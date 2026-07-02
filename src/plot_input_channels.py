#!/usr/bin/env python3
"""
Plot the actual model inputs that data.py produces -- a sampled training tile
after normalization, mask derivation, cloud-distance, MUR fill, and target-day
masking. Like plot_cube_map, but for any of the 21 UNet input channels (and the
supervision masks/targets the loss uses).

    python src/plot_input_channels.py --config configs/config.yaml --list
    python src/plot_input_channels.py --config configs/config.yaml --channel eco_mask
    python src/plot_input_channels.py --config configs/config.yaml --channel eco_sst --time-index 8
    python src/plot_input_channels.py --config configs/config.yaml --channel eco_target --aoi hood_canal
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import OceansrTileDataset, CHANNELS, MASK_CH, DIST_CH   # noqa: E402

TARGETS = ["eco_target", "eco_mask", "lst_target", "lst_mask"]    # data.py output dict (H,W)
BINARY = MASK_CH | {"eco_mask", "lst_mask"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--channel", help="input channel or target name (see --list)")
    ap.add_argument("--time-index", type=int, default=None,
                    help="day within the window (default = masked target day)")
    ap.add_argument("--aoi", default=None, help="restrict the sample to one AOI")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0, help="which sample to draw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.channel:
        print("input channels (21):")
        for i, c in enumerate(CHANNELS):
            print(f"  {i:2d} {c}")
        print("target/mask outputs:", TARGETS)
        return

    cfg = yaml.safe_load(open(args.config))
    ds = OceansrTileDataset(cfg, split=args.split, seed=args.seed)
    if args.aoi:
        ds.tiles = ds.tiles[ds.tiles["aoi"] == args.aoi].reset_index(drop=True)
        if len(ds.tiles) == 0:
            raise SystemExit(f"AOI '{args.aoi}' not in split '{args.split}'")
    s = ds[args.seed]
    tp = int(s["target_pos"])
    name = args.channel

    if name in CHANNELS:
        x = s["x"].numpy()                                   # (C,T,H,W)
        ti = tp if args.time_index is None else args.time_index
        img = x[CHANNELS.index(name), ti]
        sub = f"t={ti}" + ("  (masked target day)" if ti == tp else "")
    elif name in TARGETS:
        img = s[name].numpy()                                # (H,W) at target day
        sub = f"target day t={tp}"
    else:
        raise SystemExit(f"unknown channel '{name}'. Use --list.")

    if name in BINARY:
        cmap, vmin, vmax = "gray_r", 0, 1
    elif name in DIST_CH:
        cmap, vmin, vmax = "viridis", 0, 1
    else:                                                    # normalized continuous
        f = img[np.isfinite(img)]
        vmin, vmax = (np.percentile(f, [2, 98]) if f.size else (None, None))
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.85, label=name)
    ax.set_xticks([]); ax.set_yticks([])
    f = img[np.isfinite(img)]
    stats = (f"min {f.min():.3g} max {f.max():.3g} mean {f.mean():.3g} "
             f"nz {np.count_nonzero(img)/img.size:.0%}") if f.size else "empty"
    ax.set_title(f"{name}  |  {s['aoi']}  |  {sub}  |  mode={s['mode']}\n{stats}", fontsize=10)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"input_{s['aoi']}_{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  | {stats}")


if __name__ == "__main__":
    main()