#!/usr/bin/env python3
"""
Visualize OCEANSR model predictions on masked-day samples.

For each sample the model never sees the target day's high-res obs; this plots,
for that held-out day, the MUR backbone, the model's predicted skin SST, the true
high-res obs, and the error (pred - truth), all in Kelvin.

    python src/visualize_prediction.py --config configs/config.yaml \
        --checkpoint results/checkpoints/best.pt --split val --aoi hood_canal --n 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet3D                                  # noqa: E402
from data import OceansrTileDataset, num_input_channels, MUR_INDEX, CHANNELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="results/checkpoints/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--aoi", default=None, help="restrict to one AOI (must be in the split)")
    ap.add_argument("--n", type=int, default=4, help="number of samples to draw")
    ap.add_argument("--out", default="results/predictions.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=device)
    sst_mean, sst_std = float(ck["sst_mean"]), float(ck["sst_std"])
    model = UNet3D(in_channels=num_input_channels(cfg),
                   base_width=int(mcfg.get("base_width", 48)),
                   depth=int(mcfg.get("depth", 3)), mur_index=MUR_INDEX).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    b_lst = float(model.sensor_offset("lst").detach().cpu().item()
                  if hasattr(model.sensor_offset("lst"), "item") else 0.0)

    ds = OceansrTileDataset(cfg, split=args.split, require_tiles=False)
    if args.aoi:
        ds.tiles = ds.tiles[ds.tiles["aoi"] == args.aoi].reset_index(drop=True)
    if len(ds.tiles) == 0:
        raise SystemExit(f"no tiles in split '{args.split}'"
                         + (f" for AOI '{args.aoi}'" if args.aoi else "")
                         + " -- assemble/hold out an AOI first")

    LAND_IDX = CHANNELS.index("landmask")
    rng = np.random.default_rng(0)

    def draw_sample():
        """Random supervised sample with diagnostics (mirrors the training sampler)."""
        for _ in range(128):
            row = ds.tiles.iloc[int(rng.integers(len(ds.tiles)))]
            aoi, y0, x0 = row["aoi"], int(row["y0"]), int(row["x0"])
            obs_days, n_days = ds._obs_days(aoi)
            if len(obs_days) == 0 or n_days < ds.T:
                continue
            td = int(rng.choice(obs_days))
            mode = (ds.modes[0] if len(ds.modes) == 1
                    else ds.modes[int(np.searchsorted(np.cumsum(ds.mode_p), rng.random()))])
            s = ds.build_sample(aoi, td, mode, y0=y0, x0=x0, H=ds.S, W=ds.S,
                                blank_target=True, enforce_min=True, return_diagnostics=True)
            if s is not None:
                return s
        raise SystemExit("could not draw a supervised sample -- loosen filters / check split")

    denorm = lambda a: a * sst_std + sst_mean
    v_cmap = plt.cm.viridis.copy(); v_cmap.set_bad("#e8e8e8")     # land/no-data -> light gray
    e_cmap = plt.cm.RdBu_r.copy();  e_cmap.set_bad("#e8e8e8")

    fig, axes = plt.subplots(args.n, 4, figsize=(14, 3.4 * args.n), squeeze=False)
    for r in range(args.n):
        s = draw_sample()
        with torch.no_grad():
            pred = model(s["x"].unsqueeze(0).to(device))      # (1,1,T,H,W)
        tp = int(s["target_pos"])
        pred_day = pred[0, 0, tp].cpu().numpy()
        mur_day = s["x"][MUR_INDEX, tp].numpy()
        water = s["x"][LAND_IDX, tp].numpy() < 0.5            # landmask 1=land -> water where <0.5
        eco_t, eco_m = s["eco_target"].numpy(), s["eco_mask"].numpy()
        lst_t, lst_m = s["lst_target"].numpy(), s["lst_mask"].numpy()
        # truth in the prediction's (anchor) frame: lst shifted by -b_lst
        truth = np.where(eco_m > 0, eco_t, np.where(lst_m > 0, lst_t - b_lst, np.nan))
        kept = (eco_m > 0) | (lst_m > 0)                      # pixels actually supervised

        # "thrown out": an obs existed over water but was excluded from the loss
        present = ((s["eco_present"].numpy() > 0.5) | (s["lst_present"].numpy() > 0.5)) & water
        thrown = present & ~kept

        wmask = lambda a: np.where(water, a, np.nan)          # show water pixels only
        murK, predK = wmask(denorm(mur_day)), wmask(denorm(pred_day))
        truthK = wmask(denorm(truth))
        errK = np.where(kept, denorm(pred_day) - denorm(truth), np.nan)
        rmse = float(np.sqrt(np.nanmean(errK ** 2))) if kept.any() else float("nan")

        finite = np.concatenate([murK[np.isfinite(murK)], truthK[np.isfinite(truthK)]])
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (np.nan, np.nan))
        emax = np.nanmax(np.abs(errK)) if kept.any() else 1.0

        panels = [("MUR (input)", murK, v_cmap, vmin, vmax, False),
                  ("prediction", predK, v_cmap, vmin, vmax, True),
                  ("truth (held-out)", truthK, v_cmap, vmin, vmax, True),
                  (f"error  RMSE={rmse:.2f}K", errK, e_cmap, -emax, emax, False)]
        for c, (title, img, cmap, lo, hi, shade) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            if shade and thrown.any():                        # red = obs dropped from the loss
                red = np.zeros((*thrown.shape, 4), "float32")
                red[..., 0] = 1.0; red[..., 3] = np.where(thrown, 0.55, 0.0)
                ax.imshow(red)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
        axes[r, 0].set_ylabel(f"{s['aoi']}\n{s['mode']}  (red=dropped)", fontsize=9)

    fig.suptitle(f"OCEANSR predictions ({args.split})  |  {Path(args.checkpoint).name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
