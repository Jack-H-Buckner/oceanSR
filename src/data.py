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
  21 cloud_cover  (HRRR total cloud cover forcing)
  22 eco_finite   23 eco_gapdist  (ECOSTRESS presence mask + dist-to-nearest-gap, derived)
  24 lst_finite   25 lst_gapdist  (Landsat presence mask + dist-to-nearest-gap, derived)
  26 landcover_water  (ESA WorldCover water mask, 1=water; filterable in the loss)
  27 depth_p25   28 depth_p75  (sub-grid depth percentiles from CUDEM, static)
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
    "cloud_cover",                       # HRRR total cloud cover forcing (gap-free, %->0..1)
    "eco_finite", "eco_gapdist",         # ECOSTRESS presence mask + distance-to-nearest-gap, derived
    "lst_finite", "lst_gapdist",         # Landsat   presence mask + distance-to-nearest-gap, derived
    "landcover_water",                   # ESA WorldCover water mask (1=water); filterable in loss
    "depth_p25", "depth_p75",            # sub-grid depth percentiles (bathy variability), static
]
MUR_INDEX = CHANNELS.index("mur_sst")          # == 4
CUBE_VAR = {  # output channel -> cube variable name (derived clouddist not listed)
    "eco_sst": "eco_sst", "eco_mask": "eco_valid", "lst_sst": "lst_sst",
    "lst_mask": "lst_valid", "mur_sst": "mur_sst", "airtemp": "airtemp",
    "wind_u": "wind_u", "wind_v": "wind_v", "wind_speed": "wind_speed",
    "swrad": "swrad", "tide": "tide", "depth": "depth", "landmask": "landmask",
    "doy_sin": "doy_sin", "doy_cos": "doy_cos", "eco_hour": "eco_hour",
    "lst_hour": "lst_hour", "eco_cloud": "eco_cloud", "lst_cloud": "lst_cloud",
    "cloud_cover": "cloud_cover", "landcover_water": "landcover_water",
    "depth_p25": "depth_p25", "depth_p75": "depth_p75",
}
SST_CH = {"eco_sst", "lst_sst", "mur_sst"}
MASK_CH = {"eco_mask", "lst_mask", "landmask", "eco_cloud", "lst_cloud",
           "eco_finite", "lst_finite", "landcover_water"}   # 0/1, no z-score
RAW_CH = {"doy_sin", "doy_cos"}                # already ~[-1,1]
HOUR_CH = {"eco_hour", "lst_hour"}             # scale by /24
DIST_CH = {"eco_clouddist", "lst_clouddist", "eco_gapdist", "lst_gapdist"}  # derived, 0..1
PCT_CH = {"cloud_cover"}                        # 0..100 % -> /100 to 0..1
DERIVED_CH = DIST_CH | {"eco_finite", "lst_finite"}   # not read from cube; built after the loop
CLOUDDIST_MAX_PX = 30.0                         # cap distance-to-cloud (=3 km at 100 m)

# comparison operators allowed in loss_pixel_filters
_OPS = {">=": np.greater_equal, "<=": np.less_equal, ">": np.greater,
        "<": np.less, "==": np.equal, "!=": np.not_equal}


def _cloud_distance(cloud):
    """Per-day normalized distance-to-nearest-cloud (0 at cloud, 1 far away)."""
    from scipy.ndimage import distance_transform_edt
    out = np.empty_like(cloud, dtype="float32")
    for d in range(cloud.shape[0]):
        cld = cloud[d] > 0.5
        dd = distance_transform_edt(~cld) if cld.any() else np.full(cld.shape, CLOUDDIST_MAX_PX)
        out[d] = (np.clip(dd, 0, CLOUDDIST_MAX_PX) / CLOUDDIST_MAX_PX).astype("float32")
    return out


# Channels the config can switch off. They MUST stay at the END of CHANNELS so
# dropping them can't shift any other channel's index (MUR_INDEX, cloud_cover...).
OPTIONAL_CHANNELS = {
    "use_depth_percentiles": ["depth_p25", "depth_p75"],   # model.use_depth_percentiles
}


def active_channels(cfg=None) -> list:
    """CHANNELS minus any optional groups disabled in cfg['model']."""
    mcfg = (cfg or {}).get("model", {})
    drop = set()
    for flag, chans in OPTIONAL_CHANNELS.items():
        if not mcfg.get(flag, True):
            drop.update(chans)
    return [c for c in CHANNELS if c not in drop]


def num_input_channels(cfg=None) -> int:
    return len(active_channels(cfg)) if cfg is not None else len(CHANNELS)


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
    def __init__(self, cfg: dict, split: str = "train", seed: int = 0, require_tiles: bool = True):
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
        self.sensors = [s.lower() for s in tcfg.get("sensors", ["eco", "lst"])]  # supervised
        self.channels = active_channels(cfg)      # config-selected input channels
        self.n_channels = len(self.channels)
        self.valid_from_sst = bool(tcfg.get("valid_from_sst", False))
        self.derived_mur_filter = bool(tcfg.get("derived_mur_filter", True))
        self.mur_thr = float(cfg.get("assembler", {}).get("mur_cloud_threshold_k", 5.0))
        # physical-plausibility gate (Kelvin) so fill/sentinel/out-of-range pixels
        # don't count as observations. Set null/[] to disable. Default coastal SST.
        rng = tcfg.get("sst_valid_range_k", [270.0, 320.0])
        self.sst_valid_range = (float(rng[0]), float(rng[1])) if rng else None
        # cloud gate on supervision: drop target pixels where HRRR total cloud
        # cover (%) exceeds this, so cloudy target days get resampled. null = off.
        ct = tcfg.get("cloud_loss_threshold_pct", None)
        self.cloud_loss_thr = (float(ct) / 100.0) if ct is not None else None   # 0..1
        # Stage A: example selection (scene cloud cover + per-sensor valid fraction)
        sc = tcfg.get("select_max_cloud_cover_pct", None)
        self.select_max_cloud = (float(sc) / 100.0) if sc is not None else None  # 0..1
        # per-sensor coverage: ECOSTRESS judged on FINITE (presence) fraction,
        # Landsat on VALID fraction; combine = any. 0 disables that sensor's check.
        self.select_min_finite_frac_eco = float(tcfg.get("select_min_finite_frac_eco", 0.0))
        self.select_min_valid_frac_lst = float(tcfg.get("select_min_valid_frac_lst", 0.0))
        # Stage B: per-pixel, per-sensor loss filters
        self.loss_pixel_filters = self._parse_filters(tcfg.get("loss_pixel_filters", []) or [])
        self._bypass_selection = False        # set True in the __getitem__ fallback

        # split AOIs: val_aois held out
        val_aois = set(tcfg.get("val_aois", []))
        keep = (lambda a: a in val_aois) if split == "val" else (lambda a: a not in val_aois)

        # tiles for this split
        tiles = pd.read_csv(train_dir / "tile_index.csv")
        tiles = tiles[tiles["aoi"].apply(keep)].reset_index(drop=True)
        # Drop AOIs that can't yield a fixed S×S, T-day tile (grid smaller than the
        # tile, or fewer days than the window). tile_index can list a (0,0) tile for
        # an AOI whose grid is < tile_size; build_sample would return None for it and
        # the default collate then crashes ('NoneType' is not subscriptable).
        if len(tiles):
            usable, dropped = [], {}
            for a in tiles["aoi"].unique():
                try:
                    zd = xr.open_zarr(self.zarr_dir / f"{a}.zarr")
                    ny, nx, nt = int(zd.sizes["y"]), int(zd.sizes["x"]), int(zd.sizes["time"])
                    zd.close()
                except Exception as exc:
                    dropped[a] = f"open failed: {exc}"; continue
                if ny < self.S or nx < self.S:
                    dropped[a] = f"grid {ny}x{nx} < tile_size {self.S}"
                elif nt < self.T:
                    dropped[a] = f"{nt} days < t_window {self.T}"
                else:
                    usable.append(a)
            if dropped:
                print(f"[data:{split}] skipping {len(dropped)} AOI(s) too small for "
                      f"{self.S}px/{self.T}d tiles: {dropped} "
                      f"(lower model.tile_size to include them)")
            tiles = tiles[tiles["aoi"].isin(usable)].reset_index(drop=True)
        if len(tiles) == 0 and split == "train" and require_tiles:
            raise SystemExit("no tiles for split 'train' in tile_index.csv -- "
                             "are any AOIs assembled, and not all listed in val_aois? "
                             "(viewers pass require_tiles=False to inspect any AOI)")
        self.tiles = tiles
        # length is in SAMPLES; with drop_last the loader yields exactly
        # steps_per_epoch (val_steps) BATCHES = optimizer steps.
        bs = int(tcfg.get("batch_size", 12))
        if len(tiles) == 0:                         # empty val split -> loader skipped
            self.length = 0
        elif split == "train":
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
            water = (ds["landmask"] == 0) if "landmask" in ds else None
            for s in self.sensors:
                if self.valid_from_sst and f"{s}_sst" in ds:
                    m = np.isfinite(ds[f"{s}_sst"])
                    if water is not None:
                        m = m & water
                    has |= (m.sum(dim=("y", "x")) > 0).values
                elif f"{s}_valid" in ds:
                    has |= (ds[f"{s}_valid"].sum(dim=("y", "x")) > 0).values
            self._obs[aoi] = (np.where(has)[0], ds.sizes["time"])
        return self._obs[aoi]

    def __len__(self):
        return self.length

    # -- core ------------------------------------------------------------- #
    # -- filters ---------------------------------------------------------- #
    def _parse_filters(self, specs):
        """Validate loss_pixel_filters specs -> list of {channel, op_fn, value, sensor}."""
        out = []
        for f in specs:
            ch = f["channel"]
            if ch not in CHANNELS:
                raise SystemExit(f"loss_pixel_filters: unknown channel '{ch}'")
            op = f.get("op", ">=")
            if op not in _OPS:
                raise SystemExit(f"loss_pixel_filters: bad op '{op}' (use {list(_OPS)})")
            sensor = f.get("sensor")
            if sensor is None:
                pre = ch.split("_")[0]
                sensor = pre if pre in ("eco", "lst") else "both"
            out.append({"channel": ch, "op": _OPS[op], "value": float(f["value"]),
                        "sensor": sensor, "desc": f"{ch}{op}{f['value']}"})
        return out

    def _selection_ok(self, chans, tp, water, lst_valid):
        """Stage A: accept/reject the whole target day. ECOSTRESS coverage uses its
        FINITE (presence) fraction over water; Landsat uses its VALID fraction.
        Combine = any (accept if either sensor clears its threshold)."""
        if self.select_max_cloud is not None and water.any():
            if float(chans["cloud_cover"][tp][water].mean()) > self.select_max_cloud:
                return False                                    # scene too cloudy
        nw = int(water.sum())
        if nw == 0:
            return True
        checks = []
        if self.select_min_finite_frac_eco > 0:
            ef = float(((chans["eco_finite"][tp] > 0.5) & water).sum()) / nw
            checks.append(ef >= self.select_min_finite_frac_eco)
        if self.select_min_valid_frac_lst > 0:
            lf = float(lst_valid.sum()) / nw
            checks.append(lf >= self.select_min_valid_frac_lst)
        return (not checks) or any(checks)                      # any-sensor pass

    def _apply_loss_filters(self, chans, tp, eco_m, lst_m):
        """Stage B: per-sensor channel-rule masks AND'd into the supervision masks.
        eco_* rules touch only the ECOSTRESS mask, lst_* only Landsat (sensor='both'
        touches both). Evaluated on the UN-blanked target-day slice."""
        masks = {"eco": eco_m, "lst": lst_m}
        for f in self.loss_pixel_filters:
            keep = f["op"](chans[f["channel"]][tp], f["value"]).astype("float32")
            for s in (("eco", "lst") if f["sensor"] == "both" else (f["sensor"],)):
                masks[s] = masks[s] * keep
        return masks["eco"], masks["lst"]

    def _fetch(self, dsw, var):
        T, H, W = dsw.sizes["time"], dsw.sizes["y"], dsw.sizes["x"]
        if var not in dsw:
            return np.zeros((T, H, W), dtype="float32")
        v = np.asarray(dsw[var].values, dtype="float32")
        if v.ndim == 3:          # (T,H,W)
            return v
        if v.ndim == 2:          # static (H,W)
            return np.broadcast_to(v, (T, H, W)).copy()
        return np.broadcast_to(v[:, None, None], (T, H, W)).copy()  # (T,)

    def build_sample(self, aoi, target_day, mode="interior", y0=0, x0=0,
                     H=None, W=None, blank_target=True, enforce_min=False,
                     return_diagnostics=False):
        """Build one processed (C,T,H,W) sample for explicit params. This is the
        SINGLE code path shared by training (_try_sample, on 128-px tiles) and the
        input-viewer script (which can ask for the full AOI extent). Returns None
        only on an off-grid tile, or too-few target pixels when enforce_min=True.

        H/W default to the full cube extent; pass H=W=self.S for a training tile."""
        ds = self._zarr(aoi)
        n_days = ds.sizes["time"]
        Hs = ds.sizes["y"] if H is None else H
        Ws = ds.sizes["x"] if W is None else W
        start, tp = make_window(target_day, n_days, self.T, mode)
        dsw = ds.isel(time=slice(start, start + self.T),
                      y=slice(y0, y0 + Hs), x=slice(x0, x0 + Ws)).load()
        if dsw.sizes["y"] != Hs or dsw.sizes["x"] != Ws:
            return None  # tile runs off the grid edge
        T, Hs, Ws = dsw.sizes["time"], dsw.sizes["y"], dsw.sizes["x"]

        mur_n = (self._fetch(dsw, "mur_sst") - self.sst_mean) / self.sst_std
        mur_n = np.nan_to_num(mur_n, nan=0.0)

        chans = {}
        sst_finite = {}
        for name in CHANNELS:
            if name in DERIVED_CH:
                continue                                      # derived after the loop
            if self.valid_from_sst and name in ("eco_mask", "lst_mask"):
                chans[name] = np.zeros((T, Hs, Ws), "float32")  # set below
                continue                                      # never read the stored *_valid
            if name == "landcover_water" and "landcover_water" not in dsw:
                chans[name] = np.ones((T, Hs, Ws), "float32")  # absent -> all water (no-op filter)
                continue
            arr = self._fetch(dsw, CUBE_VAR[name])
            if name in SST_CH:
                fin = np.isfinite(arr)
                if self.sst_valid_range is not None:          # exclude fill/sentinel/junk
                    lo, hi = self.sst_valid_range
                    fin = fin & (arr >= lo) & (arr <= hi)
                a = (arr - self.sst_mean) / self.sst_std
                chans[name] = np.where(fin, a, mur_n)         # fill gaps with MUR
                sst_finite[name] = fin
            elif name in MASK_CH:
                chans[name] = np.nan_to_num(arr, nan=0.0)      # 0/1 (masks, cloud)
            elif name in HOUR_CH:
                chans[name] = np.nan_to_num(arr / 24.0, nan=0.0)
            elif name in PCT_CH:
                chans[name] = np.nan_to_num(arr / 100.0, nan=0.0)   # cloud cover %->0..1
            elif name in RAW_CH:
                chans[name] = np.nan_to_num(arr, nan=0.0)
            else:
                m, s = self.norm.get(CUBE_VAR[name], (0.0, 1.0))
                chans[name] = np.nan_to_num((arr - m) / s, nan=0.0)

        # cloud-proximity, derived per sensor from the binary cloud channel
        chans["eco_clouddist"] = _cloud_distance(chans["eco_cloud"])
        chans["lst_clouddist"] = _cloud_distance(chans["lst_cloud"])

        # per-sensor presence: 1 where a real (range-gated finite) obs exists, else 0.
        # *_gapdist = distance to the nearest MISSING (not-finite) pixel: 0 at a gap,
        # growing deeper into observed regions. _cloud_distance is generic
        # distance-to-True, so we feed it the inverted (missing) mask.
        chans["eco_finite"] = sst_finite["eco_sst"].astype("float32")
        chans["eco_gapdist"] = _cloud_distance(1.0 - chans["eco_finite"])
        chans["lst_finite"] = sst_finite["lst_sst"].astype("float32")
        chans["lst_gapdist"] = _cloud_distance(1.0 - chans["lst_finite"])

        # optionally derive the high-res valid mask from finite SST over water,
        # bypassing the stored *_valid (use when *_valid is broken/empty). The MUR
        # cold-deviation filter is re-applied on the fly so cloud-cold pixels are
        # still dropped (QC can't be reproduced here -- it's not in the cube).
        if self.valid_from_sst:
            water = chans["landmask"] < 0.5                   # landmask 1=land
            eco_ok = sst_finite["eco_sst"] & water
            lst_ok = sst_finite["lst_sst"] & water
            if self.derived_mur_filter and self.mur_thr > 0:
                # channels are normalized -> convert deviation back to Kelvin
                eco_dev = (chans["mur_sst"] - chans["eco_sst"]) * self.sst_std
                lst_dev = (chans["mur_sst"] - chans["lst_sst"]) * self.sst_std
                eco_ok &= ~(eco_dev > self.mur_thr)
                lst_ok &= ~(lst_dev > self.mur_thr)
            chans["eco_mask"] = eco_ok.astype("float32")
            chans["lst_mask"] = lst_ok.astype("float32")

        # diagnostic: the MUR cold-deviation "cloud" flag -- pixels a sensor reads
        # > mur_thr K colder than the MUR backbone (only where it has a real obs).
        # This is exactly what derived_mur_filter drops; exposed for visualization.
        diag = {}
        if return_diagnostics:
            for sname in ("eco", "lst"):
                dev = (chans["mur_sst"] - chans[f"{sname}_sst"]) * self.sst_std
                cold = (dev > self.mur_thr) & sst_finite[f"{sname}_sst"]
                diag[f"{sname}_mur_cold"] = torch.from_numpy(cold.astype("float32"))
                # presence at the target day (before blanking): 1 where a real obs exists
                diag[f"{sname}_present"] = torch.from_numpy(
                    (chans[f"{sname}_finite"][tp] > 0.5).astype("float32"))

        # targets = TRUE high-res at the target day (before masking)
        eco_t, eco_m = chans["eco_sst"][tp].copy(), chans["eco_mask"][tp].copy()
        lst_t, lst_m = chans["lst_sst"][tp].copy(), chans["lst_mask"][tp].copy()
        # ---- Stage A: example selection (reject the whole target day) ----
        # uses RAW valid coverage (before the Stage-B pixel filters). Skipped when
        # enforce_min is False (viewer) or during the __getitem__ relaxed fallback.
        water2d = chans["landmask"][tp] < 0.5
        sel_ok = self._selection_ok(chans, tp, water2d, lst_m)
        if return_diagnostics:
            diag["selection_ok"] = bool(sel_ok)
        if enforce_min and not self._bypass_selection and not sel_ok:
            return None
        # ---- Stage B: per-pixel loss inclusion ----
        # global HRRR cloud-cover pixel gate (both sensors), then per-sensor rules.
        if self.cloud_loss_thr is not None:
            clear = chans["cloud_cover"][tp] <= self.cloud_loss_thr
            eco_m = eco_m * clear
            lst_m = lst_m * clear
        eco_m, lst_m = self._apply_loss_filters(chans, tp, eco_m, lst_m)
        if enforce_min and eco_m.sum() + lst_m.sum() < self.min_target_px:
            return None  # too few supervised pixels after filtering -> resample

        # whole-day mask: blank EVERYTHING from the masked sensor on the target day
        # (no overpass -> no SST, no cloud, no proximity; avoids target leakage)
        if blank_target:
            for s in ("eco", "lst"):
                chans[f"{s}_sst"][tp] = mur_n[tp]
                chans[f"{s}_mask"][tp] = 0.0
                chans[f"{s}_cloud"][tp] = 0.0
                chans[f"{s}_clouddist"][tp] = 0.0
            for s in ("eco", "lst"):
                chans[f"{s}_finite"][tp] = 0.0   # held-out day -> no obs present
                chans[f"{s}_gapdist"][tp] = 0.0  # ... every pixel is a gap -> distance 0

        x = np.stack([chans[c] for c in self.channels], axis=0).astype("float32")
        out = {
            "x": torch.from_numpy(x),                       # (C,T,H,W)
            "target_pos": tp,
            "eco_target": torch.from_numpy(eco_t),          # (H,W) normalized
            "eco_mask": torch.from_numpy(eco_m),            # (H,W) 0/1
            "lst_target": torch.from_numpy(lst_t),
            "lst_mask": torch.from_numpy(lst_m),
            "mode": mode, "aoi": aoi,
        }
        out.update(diag)                                    # *_mur_cold (T,H,W) if requested
        return out

    def _sample_tile(self, i, rng):
        row = self.tiles.iloc[i]
        aoi, y0, x0 = row["aoi"], int(row["y0"]), int(row["x0"])
        obs_days, n_days = self._obs_days(aoi)
        if len(obs_days) == 0 or n_days < self.T:
            return None
        target_day = int(rng.choice(obs_days))
        mode = self.modes[rng.integers(len(self.modes)) if len(self.modes) == 1
                          else np.searchsorted(np.cumsum(self.mode_p), rng.random())]
        return self.build_sample(aoi, target_day, mode, y0=y0, x0=x0,
                                 H=self.S, W=self.S, blank_target=True, enforce_min=True)

    def _try_sample(self, rng):
        return self._sample_tile(int(rng.integers(len(self.tiles))), rng)

    def __getitem__(self, idx):
        wid = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + idx + (wid.id * 100003 if wid else 0))
        for _ in range(self.max_retries):
            out = self._try_sample(rng)
            if out is not None:
                return out
        # relaxed fallback: drop the pixel floor AND Stage-A selection, keep trying,
        # then deterministically scan tiles so __getitem__ never returns None (which
        # would crash the collate). Only raises if NOTHING is trainable.
        old, self.min_target_px = self.min_target_px, 0
        self._bypass_selection = True
        try:
            for _ in range(max(32, self.max_retries * 4)):
                out = self._try_sample(rng)
                if out is not None:
                    return out
            for i in range(len(self.tiles)):
                out = self._sample_tile(i, rng)
                if out is not None:
                    return out
        finally:
            self._bypass_selection = False
            self.min_target_px = old
        raise RuntimeError(
            "OceansrTileDataset: could not build any valid sample. Do the assembled "
            "AOIs have observed days (sensors/valid_from_sst) and grids >= tile_size?")


def make_loader(cfg, split="train", seed=0):
    from torch.utils.data import DataLoader
    tcfg = cfg.get("train", {})
    ds = OceansrTileDataset(cfg, split=split, seed=seed)
    if len(ds.tiles) == 0:           # empty val split -> no loader (validation skipped)
        return None
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
