#!/usr/bin/env python3
"""
Visualize how a square training tile interacts with an AOI's scale: plots the
AOI bathymetry, overlays the tiling grid, and highlights one tile.

    python src/plot_tile_over_bathy.py --aoi hood_canal --tile 128 --res 100
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--bathy-dir", default="data/BATHYMETRY/aligned")
    ap.add_argument("--tile", type=int, default=128, help="tile side in pixels")
    ap.add_argument("--stride", type=int, default=None, help="default = tile (non-overlapping)")
    ap.add_argument("--res", type=float, default=100.0, help="grid resolution in metres")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    stride = args.stride or args.tile

    f = Path(args.bathy_dir) / args.aoi / f"{args.aoi}.nc"
    if not f.exists():
        raise SystemExit(f"no bathymetry at {f}")
    ds = xr.open_dataset(f)
    depth = ds["depth"].values.astype("float32")          # m, 0 on land
    H, W = depth.shape
    land = depth <= 0
    disp = np.where(land, np.nan, depth)                  # show water only
    km = args.res / 1000.0
    Wkm, Hkm, tile_km = W * km, H * km, args.tile * km

    fig, ax = plt.subplots(figsize=(max(6, Wkm / 4), max(5, Hkm / 4)))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#d9d2c5")  # land = tan
    im = ax.imshow(disp, cmap=cmap, origin="upper",
                   extent=[0, Wkm, 0, Hkm], aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="depth (m)")

    # tiling grid (light) + count tiles whose origin fits
    nx = ny = 0
    for y0 in range(0, max(H - args.tile + 1, 1), stride):
        ny += 1
        for x0 in range(0, max(W - args.tile + 1, 1), stride):
            if y0 == 0:
                nx += 1
            ax.add_patch(Rectangle((x0 * km, Hkm - (y0 + args.tile) * km),
                                   tile_km, tile_km, fill=False,
                                   edgecolor="white", lw=0.6, alpha=0.7))

    # highlight the centre tile
    cy = max((H - args.tile) // 2, 0)
    cx = max((W - args.tile) // 2, 0)
    ax.add_patch(Rectangle((cx * km, Hkm - (cy + args.tile) * km), tile_km, tile_km,
                           fill=False, edgecolor="red", lw=2.2,
                           label=f"{args.tile}² tile = {tile_km:.1f} km"))

    ax.set_xlabel("km"); ax.set_ylabel("km")
    ax.set_title(f"{args.aoi}:  {Wkm:.1f} × {Hkm:.1f} km grid "
                 f"({W}×{H}px)  |  tile {tile_km:.1f} km  |  ~{nx}×{ny} tiles")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"tile_over_bathy_{args.aoi}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"wrote {p}  | grid {W}x{H}px = {Wkm:.1f}x{Hkm:.1f} km | "
          f"tile {args.tile}px = {tile_km:.1f} km | water frac {1 - land.mean():.2f}")


if __name__ == "__main__":
    main()