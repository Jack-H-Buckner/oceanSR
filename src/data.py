#!/usr/bin/env python3
"""
OCEANSR -- training Dataset.

Turns the per-AOI Zarr cubes + tile_index.csv + norm_stats.json into the
(C, T, H, W) tensors UNet3D expects, with whole-day masking for the
interpolation (interior day) / nowcasting (last day) objective.

Per sample:
  1. pick a tile (aoi, y0, x0) from tile_index.csv,
  2. pick a target day that has a high-res observation, and a mask mode
     (interior -> interpolation, last -> nowcasting),
  3. read the T-day window for that tile,
  4. normalize (SST channels share one scale; forcing/depth z-scored),
  5. stash the target day's TRUE high-res obs + masks, then blank that day's
     high-res input (fill with MUR, set masks to 0),
  6. return inputs + targets for the masked-loss step.

Channel order (must match UNet3D.mur_index = 4):
  0 eco_sst  1 eco_mask  2 lst_sst  3 lst_mask  4 mur_sst
  5 airtemp  6 wind_u    7 wind_v   8 wind_speed 9 swrad 10 tide
  11 depth   12 landmask
  13 doy_sin 14 doy_cos  15 eco_hour 16 lst_hour
  17 eco_cloud 18 lst_cloud 19 eco_clouddist 20 lst_clouddist
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset

CHANNELS = [
    "eco_sst", "eco_mask", "lst_sst", "lst_mask", "mur_sst",
    "airtemp", "wind_u", "wind_v", "wind_speed", "swrad", "tide",
    "depth", "landmask",
    "doy_sin", "doy_cos", "eco_hour", "lst_hour",
    # cloud awareness (appended so mur_index stays 4):
    "eco_cloud", "lst_cloud",            # binary cloud (informational, not a hard mask)
    "eco_clouddist", "lst_clouddist",    # distance-to-cloud (contamination gradient), derived
]
MUR_INDEX = CHANNELS.index("mur_sst")          # == 4
CUBE_VAR = {  # output channel -> cube variable name (derived clouddist not listed)
    "eco_sst": "eco_sst", "eco_mask": "eco_valid", "lst_sst": "lst_sst",
    "lst_mask": "lst_valid", "mur_sst": "mur_sst", "airtemp": "airtemp",
    "wind_u": "wind_u", "wind_v": "wind_v", "wind_speed": "wind_speed",
    "swrad": "swrad", "tide": "tide", "depth": "depth", "landmask": "landmask",
    "doy_sin": "doy_sin", "doy_cos": "doy_cos", "eco_hour": "eco_hour",
    "lst_hour": "lst_hour", "eco_cloud": "eco_cloud", "lst_cloud": "lst_cloud",
}
SST_CH = {"eco_sst", "lst_sst", "mur_sst"}
MASK_CH = {"eco_mask", "lst_mask", "landmask", "eco_cloud", "lst_cloud"}  # 0/1, no z-score
RAW_CH = {"doy_sin", "doy_cos"}                # already ~[-1,1]
HOUR_CH = {"eco_hour", "lst_hour"}             # scale by /24
DIST_CH = {"eco_clouddist", "lst_clouddist"}   # derived from cloud, normalized 0..1
CLOUDDIST_MAX_PX = 30.0                         # cap distance-to-cloud (=3 km at 100 m)


def _cloud_distance(cloud):
    """Per-day normalized distance-to-nearest-cloud (0 at cloud, 1 far away)."""
    from scipy.ndimage import distance_transform_edt
    out = np.empty_like(cloud, dtype="float32")
    for d in range(cloud.shape[0]):
        cld = cloud[d] > 0.5
        dd = distance_transform_edt(~cld) if cld.any() else np.full(cld.shape, CLOUDDIST_MAX_PX)
        out[d] = (np.clip(dd, 0, CLOUDDIST_MAX_PX) / CLOUDDIST_MAX_PX).astype("float32")
    return out


def num_input_channels() -> int:
    return len(CHANNELS)


def _pool_sst_stats(stats: dict):
    """Combine mur/eco/lst per-channel stats into ONE shared SST mean/std so that
    `MUR + residual` and the per-sensor targets live in the same space."""
    n = m = s2 = 0.0
    for k in ("mur_sst", "eco_sst", "lst_sst"):
        st = stats.get(k)
        if not st:
            continue
        ni = st.get("count", 1)
        n += ni
        m += ni * st["mean"]
        s2 += ni * (st["std"] ** 2 + st["mean"] ** 2)
    if n == 0:
        return 0.0, 1.0
    mean = m / n
    var = max(s2 / n - mean ** 2, 1e-6)
    return mean, math.sqrt(var)


def make_window(target_day: int, n_days: int, T: int, mode: str):
    """Place a T-day window; return (start, target_pos). mode 'last' -> nowcasting."""
    start = target_day - (T - 1) if mode == "last" else target_day - T // 2
    start = max(0, min(start, n_days - T))
    return start, target_day - start


class OceansrTileDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train", seed: int = 0):
        self.cfg = cfg
        self.split = split
        self.seed = seed
        root = Path(cfg.get("project", {}).get("root", "."))
        train_dir = root / cfg["paths"]["data"] / cfg["paths"]["training"]
        self.zarr_dir = train_dir

        mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
        self.T = int(mcfg.get("t_window", 16))
        self.S = int(mcfg.get("tile_size", 128))
        mix = tcfg.get("mask_mode_mix", {"interior": 0.6, "last": 0.4})
        self.modes = list(mix.keys())
        self.mode_p = np.array([mix[m] for m in self.modes], dtype=float)
        self.mode_p /= self.mode_p.sum()
        self.max_retries = int(tcfg.get("sample_retries", 8))
        self.min_target_px = int(tcfg.get("min_target_px", 16))

        # split AOIs: val_aois held out
        val_aois = set(tcfg.get("val_aois", []))
        keep = (lambda a: a in val_aois) if split == "val" else (lambda a: a not in val_aois)

        # tiles for this split
        tiles = pd.read_csv(train_dir / "tile_index.csv")
        tiles = tiles[tiles["aoi"].apply(keep)].reset_index(drop=True)
        if len(tiles) == 0:
            raise SystemExit(f"no tiles for split '{split}' in tile_index.csv")
        self.tiles = tiles
        # length is in SAMPLES; with drop_last the loader yields exactly
        # steps_per_epoch (val_steps) BATCHES = optimizer steps.
        bs = int(tcfg.get("batch_size", 12))
        if split == "train":
            self.length = int(tcfg.get("steps_per_epoch", 1000)) * bs
        else:
            self.length = int(tcfg.get("val_steps", 256)) * bs

        # normalization
        stats = json.load(open(train_dir / "norm_stats.json"))
        self.sst_mean, self.sst_std = _pool_sst_stats(stats)
        self.norm = {k: (v["mean"], max(v["std"], 1e-6)) for k, v in stats.items()}

        self._ds = {}          # aoi -> open zarr (lazy, per worker)
        self._obs = {}         # aoi -> (obs_day_indices, n_days)

    # -- lazy per-worker handles ------------------------------------------- #
    def _zarr(self, aoi):
        if aoi not in self._ds:
            self._ds[aoi] = xr.open_zarr(self.zarr_dir / f"{aoi}.zarr")
        return self._ds[aoi]

    def _obs_days(self, aoi):
        if aoi not in self._obs:
            ds = self._zarr(aoi)
            has = np.zeros(ds.sizes["time"], dtype=bool)
            for v in ("eco_valid", "lst_valid"):
                if v in ds:
                    has |= (ds[v].sum(dim=("y", "x")) > 0).values
            self._obs[aoi] = (np.where(has)[0], ds.sizes["time"])
        return self._obs[aoi]

    def __len__(self):
        return self.length

    # -- core ------------------------------------------------------------- #
    def _fetch(self, dsw, var):
        if var not in dsw:
            return np.zeros((self.T, self.S, self.S), dtype="float32")
        v = np.asarray(dsw[var].values, dtype="float32")
        if v.ndim == 3:          # (T,H,W)
            return v
        if v.ndim == 2:          # static (H,W)
            return np.broadcast_to(v, (self.T, self.S, self.S)).copy()
        return np.broadcast_to(v[:, None, None], (self.T, self.S, self.S)).copy()  # (T,)

    def _try_sample(self, rng):
        row = self.tiles.iloc[rng.integers(len(self.tiles))]
        aoi, y0, x0 = row["aoi"], int(row["y0"]), int(row["x0"])
        obs_days, n_days = self._obs_days(aoi)
        if len(obs_days) == 0 or n_days < self.T:
            return None
        target_day = int(rng.choice(obs_days))
        mode = self.modes[rng.integers(len(self.modes)) if len(self.modes) == 1
                          else np.searchsorted(np.cumsum(self.mode_p), rng.random())]
        start, tp = make_window(target_day, n_days, self.T, mode)

        ds = self._zarr(aoi)
        dsw = ds.isel(time=slice(start, start + self.T),
                      y=slice(y0, y0 + self.S), x=slice(x0, x0 + self.S)).load()
        if dsw.sizes["y"] != self.S or dsw.sizes["x"] != self.S:
            return None  # tile runs off the grid edge

        mur_n = (self._fetch(dsw, "mur_sst") - self.sst_mean) / self.sst_std
        mur_n = np.nan_to_num(mur_n, nan=0.0)

        chans = {}
        for name in CHANNELS:
            if name in DIST_CH:
                continue                                      # derived after the loop
            arr = self._fetch(dsw, CUBE_VAR[name])
            if name in SST_CH:
                a = (arr - self.sst_mean) / self.sst_std
                chans[name] = np.where(np.isfinite(a), a, mur_n)   # fill gaps with MUR
            elif name in MASK_CH:
                chans[name] = np.nan_to_num(arr, nan=0.0)      # 0/1 (masks, cloud)
            elif name in HOUR_CH:
                chans[name] = np.nan_to_num(arr / 24.0, nan=0.0)
            elif name in RAW_CH:
                chans[name] = np.nan_to_num(arr, nan=0.0)
            else:
                m, s = self.norm.get(CUBE_VAR[name], (0.0, 1.0))
                chans[name] = np.nan_to_num((arr - m) / s, nan=0.0)

        # cloud-proximity, derived per sensor from the binary cloud channel
        chans["eco_clouddist"] = _cloud_distance(chans["eco_cloud"])
        chans["lst_clouddist"] = _cloud_distance(chans["lst_cloud"])

        # targets = TRUE high-res at the target day (before masking)
        eco_t, eco_m = chans["eco_sst"][tp].copy(), chans["eco_mask"][tp].copy()
        lst_t, lst_m = chans["lst_sst"][tp].copy(), chans["lst_mask"][tp].copy()
        if eco_m.sum() + lst_m.sum() < self.min_target_px:
            return None  # not enough held-out pixels to supervise -> resample

        # whole-day mask: blank EVERYTHING from the masked sensor on the target day
        # (no overpass -> no SST, no cloud, no proximity; avoids target leakage)
        for s in ("eco", "lst"):
            chans[f"{s}_sst"][tp] = mur_n[tp]
            chans[f"{s}_mask"][tp] = 0.0
            chans[f"{s}_cloud"][tp] = 0.0
            chans[f"{s}_clouddist"][tp] = 0.0

        x = np.stack([chans[c] for c in CHANNELS], axis=0).astype("float32")
        return {
            "x": torch.from_numpy(x),                       # (C,T,H,W)
            "target_pos": tp,
            "eco_target": torch.from_numpy(eco_t),          # (H,W) normalized
            "eco_mask": torch.from_numpy(eco_m),            # (H,W) 0/1
            "lst_target": torch.from_numpy(lst_t),
            "lst_mask": torch.from_numpy(lst_m),
            "mode": mode, "aoi": aoi,
        }

    def __getitem__(self, idx):
        wid = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + idx + (wid.id * 100003 if wid else 0))
        out = None
        for _ in range(self.max_retries):
            out = self._try_sample(rng)
            if out is not None:
                return out
        # fall back to a fresh attempt without the pixel threshold
        old, self.min_target_px = self.min_target_px, 0
        out = self._try_sample(rng) or self._try_sample(rng)
        self.min_target_px = old
        return out


def make_loader(cfg, split="train", seed=0):
    from torch.utils.data import DataLoader
    tcfg = cfg.get("train", {})
    ds = OceansrTileDataset(cfg, split=split, seed=seed)
    return DataLoader(
        ds, batch_size=int(tcfg.get("batch_size", 12)),
        shuffle=(split == "train"), num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True, drop_last=(split == "train"),
        persistent_workers=bool(tcfg.get("num_workers", 4)),
    )


if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = OceansrTileDataset(cfg, split=args.split)
    print(f"channels={num_input_channels()} tiles={len(ds.tiles)} len={len(ds)}")
    s = ds[0]
    print("x", tuple(s["x"].shape), "| target_pos", s["target_pos"], "| mode", s["mode"])
    print("eco target", tuple(s["eco_target"].shape),
          "valid px:", int(s["eco_mask"].sum() + s["lst_mask"].sum()))