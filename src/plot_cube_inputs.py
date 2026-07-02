#!/usr/bin/env python3
"""
Visualize the EXACT model inputs data.py feeds the UNet, for a chosen AOI and a
target date. Builds the T-day window via OceansrTileDataset.build_sample (same
normalization, mask derivation, cloud-distance, MUR-fill, and target-day
blanking as training), then writes a folder with ONE figure per input channel.
Each figure has one panel per day in the window; the masked target day is boxed.

    python src/plot_cube_inputs.py --config configs/config.yaml --aoi hood_canal
    python src/plot_cube_inputs.py --config configs/config.yaml --aoi hood_canal \
        --date 2023-08-15 --mode last
    python src/plot_cube_inputs.py --config configs/config.yaml --aoi padilla_bay \
        --tile 0,0 --size 128 --channels eco_sst eco_mask mur_sst

Notes
- Default spatial extent is the FULL AOI grid; pass --tile y0,x0 (+ --size) to
  view a single 128-px training tile instead.
- Continuous channels share one color scale across all day-panels so they're
  comparable; masks/cloud are 0/1, distance & hour are 0..1.
- Pass --no-blank to keep the true high-res obs on the target day (off by
  default so you see what the model actually receives -- a blanked target day).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (OceansrTileDataset, make_window, CHANNELS,  # noqa: E402
                  MASK_CH, DIST_CH, HOUR_CH, PCT_CH)

MUR_COLD = {"eco_mur_cold", "lst_mur_cold"}   # derived diagnostic, 0/1
BINARY = MASK_CH | MUR_COLD            # 0/1 channels -> gray
UNIT01 = DIST_CH | HOUR_CH | PCT_CH    # already 0..1 -> viridis 0..1


def color_spec(name, vol):
    """(cmap, vmin, vmax) shared across all day panels for a channel."""
    if name in BINARY:
        return "gray_r", 0.0, 1.0
    if name in UNIT01:
        return "viridis", 0.0, 1.0
    f = vol[np.isfinite(vol)]
    if f.size == 0:
        return "viridis", None, None
    lo, hi = np.percentile(f, [2, 98])
    if lo == hi:
        lo, hi = float(f.min()), float(f.max()) or 1.0
    return ("RdBu_r" if name == "mur_sst" or name.endswith("_sst") else "viridis"), lo, hi


def plot_channel(vol, name, times, tp, out_path):
    """vol (T,H,W) -> one figure, a panel per day, target day boxed in red."""
    T = vol.shape[0]
    ncol = min(4, T)
    nrow = math.ceil(T / ncol)
    cmap, vmin, vmax = color_spec(name, vol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             squeeze=False)
    im = None
    for t in range(nrow * ncol):
        ax = axes[t // ncol][t % ncol]
        if t >= T:
            ax.axis("off")
            continue
        im = ax.imshow(vol[t], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        day = pd.Timestamp(times[t]).strftime("%m-%d")
        is_tgt = (t == tp)
        ax.set_title((f"{day}  *TARGET*" if is_tgt else day),
                     fontsize=9, color=("red" if is_tgt else "black"))
        ax.set_xticks([]); ax.set_yticks([])
        if is_tgt:
            for sp in ax.spines.values():
                sp.set_color("red"); sp.set_linewidth(2.5)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=name, location="right")
    fig.suptitle(name, fontsize=13, y=0.99)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_single(img, name, subtitle, kind, out_path):
    """One H,W panel -- used for the target-day supervision targets/masks."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if kind == "mask":
        im = ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    else:
        f = img[np.isfinite(img)]
        vmin, vmax = (np.percentile(f, [2, 98]) if f.size else (None, None))
        im = ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.85, label=name)
    ax.set_xticks([]); ax.set_yticks([])
    extra = f"  supervised px={int(np.count_nonzero(img))}" if kind == "mask" else ""
    ax.set_title(f"{name}  (target {subtitle}){extra}", fontsize=10)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", default=None, help="target date YYYY-MM-DD (default: a mid obs day)")
    ap.add_argument("--mode", default="interior", choices=["interior", "last"],
                    help="interior=interpolation window, last=nowcasting window")
    ap.add_argument("--tile", default=None, help="'y0,x0' for a single tile (default: full AOI)")
    ap.add_argument("--size", type=int, default=None, help="tile size when --tile given (default model tile_size)")
    ap.add_argument("--channels", nargs="+", default=None, help="subset of channels (default: all 21)")
    ap.add_argument("--no-blank", action="store_true", help="keep true obs on the target day")
    ap.add_argument("--no-targets", action="store_true",
                    help="skip the target-day supervision targets/masks (which reflect the cloud gate)")
    ap.add_argument("--out", default="results/inputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = OceansrTileDataset(cfg, split="train", require_tiles=False)  # tiles unused by build_sample
    zds = ds._zarr(args.aoi)
    times = pd.to_datetime(zds["time"].values)
    n_days = zds.sizes["time"]

    # resolve target day
    if args.date:
        target_day = int(np.argmin(np.abs(times.values - np.datetime64(args.date))))
        got = times[target_day].strftime("%Y-%m-%d")
        if got != args.date:
            print(f"note: {args.date} not in cube; using nearest day {got}")
    else:
        obs_days, _ = ds._obs_days(args.aoi)
        if len(obs_days) == 0:
            raise SystemExit(f"{args.aoi}: no observed days for sensors {ds.sensors}")
        target_day = int(obs_days[len(obs_days) // 2])
    date_str = times[target_day].strftime("%Y-%m-%d")

    # spatial extent
    y0 = x0 = 0
    H = W = None
    if args.tile:
        y0, x0 = (int(v) for v in args.tile.split(","))
        H = W = args.size or ds.S

    s = ds.build_sample(args.aoi, target_day, mode=args.mode, y0=y0, x0=x0,
                        H=H, W=W, blank_target=not args.no_blank, enforce_min=False,
                        return_diagnostics=True)
    if s is None:
        raise SystemExit("tile ran off the grid edge -- adjust --tile/--size")
    x = s["x"].numpy()                                    # (C,T,H,W)
    tp = int(s["target_pos"])
    start, _ = make_window(target_day, n_days, ds.T, args.mode)
    wtimes = times[start:start + ds.T]

    names = args.channels or ds.channels
    tag = "noblank" if args.no_blank else "blanked"
    out_dir = Path(args.out) / f"{args.aoi}_{date_str}_{args.mode}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.aoi} | target {date_str} (day {target_day}, mode {args.mode}, "
          f"window {wtimes[0].strftime('%m-%d')}..{wtimes[-1].strftime('%m-%d')}) | "
          f"extent {x.shape[2]}x{x.shape[3]} | {len(names)} channels")
    for ci, name in enumerate([n for n in names if n in ds.channels]):
        idx = ds.channels.index(name)
        plot_channel(x[idx], name, wtimes, tp, out_dir / f"{idx:02d}_{name}.png")

    # MUR cold-deviation "cloud" flag per sensor (what derived_mur_filter drops):
    # 1 where the sensor is > mur_cloud_threshold_k colder than MUR over a real obs.
    want = set(args.channels) if args.channels else None
    for dname in ("eco_mur_cold", "lst_mur_cold"):
        if dname in s and (want is None or dname in want):
            plot_channel(s[dname].numpy(), dname, wtimes, tp, out_dir / f"mur_{dname}.png")

    # target-day supervision: the masks here reflect the cloud-loss gate
    # (train.cloud_loss_threshold_pct) and valid_from_sst / MUR filtering.
    if not args.no_targets:
        for tname, kind in (("eco_target", "sst"), ("eco_mask", "mask"),
                            ("lst_target", "sst"), ("lst_mask", "mask")):
            plot_single(s[tname].numpy(), tname, date_str, kind,
                        out_dir / f"tgt_{tname}.png")
        em, lm = int(s["eco_mask"].sum()), int(s["lst_mask"].sum())
        gate = "off" if ds.cloud_loss_thr is None else f"<= {ds.cloud_loss_thr*100:g}% cloud"
        print(f"supervision (cloud gate {gate}): eco_mask={em}px  lst_mask={lm}px")
    print(f"wrote {len(list(out_dir.glob('*.png')))} plots -> {out_dir}")


if __name__ == "__main__":
    main()
