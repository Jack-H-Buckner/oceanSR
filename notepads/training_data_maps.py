#!/usr/bin/env python3
"""
QC/QA map viewer for the training Zarr cubes.

Pick an AOI, a date (snaps to the nearest available time), and a variable; plots
a map (for (time,y,x) or static (y,x) vars) or a time series (for 1-D time vars),
and prints summary stats.

    python src/plot_cube_map.py --aoi hood_canal --date 2023-07-15 --var eco_sst
    python src/plot_cube_map.py --aoi hood_canal --list           # list variables
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASKISH = ("mask", "valid", "cloud", "water", "landmask")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", help="target date YYYY-MM-DD (snaps to nearest time)")
    ap.add_argument("--var", help="variable to plot")
    ap.add_argument("--zarr-dir", default="data/TRAINING")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--list", action="store_true", help="list variables and exit")
    args = ap.parse_args()

    zpath = Path(args.zarr_dir) / f"{args.aoi}.zarr"
    if not zpath.exists():
        raise SystemExit(f"no cube at {zpath}")
    ds = xr.open_zarr(zpath)

    if args.list or not args.var:
        print(f"{args.aoi}.zarr  (time={ds.sizes.get('time')}, "
              f"y={ds.sizes.get('y')}, x={ds.sizes.get('x')})")
        for v in ds.data_vars:
            print(f"  {v:12s} dims={ds[v].dims}")
        return
    if args.var not in ds:
        raise SystemExit(f"'{args.var}' not in cube. Use --list to see variables.")

    da = ds[args.var]
    dims = set(da.dims)
    cmap = args.cmap or ("gray_r" if any(k in args.var for k in MASKISH) else "viridis")
    vmin = args.vmin if args.vmin is not None else (0 if any(k in args.var for k in MASKISH) else None)
    vmax = args.vmax if args.vmax is not None else (1 if any(k in args.var for k in MASKISH) else None)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ---- 1-D time series ------------------------------------------------- #
    if da.dims == ("time",):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(da["time"].values, da.values, lw=1, color="#1f77b4")
        if args.date:
            sel = da.sel(time=pd.Timestamp(args.date), method="nearest")
            t = pd.Timestamp(sel["time"].values)
            ax.axvline(t, color="red", lw=1); ax.scatter([t], [float(sel)], color="red", zorder=5)
            ax.set_title(f"{args.aoi}: {args.var}  (marked {t.date()}, value {float(sel):.3g})")
        else:
            ax.set_title(f"{args.aoi}: {args.var}")
        ax.set_xlabel("time"); ax.set_ylabel(args.var); ax.grid(alpha=0.3)
        out = Path(args.out) / f"qc_{args.aoi}_{args.var}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"wrote {out}")
        return

    # ---- map ------------------------------------------------------------- #
    if "time" in dims:
        if not args.date:
            raise SystemExit(f"{args.var} has a time dim; pass --date")
        sel = da.sel(time=pd.Timestamp(args.date), method="nearest")
        actual = pd.Timestamp(sel["time"].values)
        img = sel.values
        req = pd.Timestamp(args.date)
        when = f"{actual.date()}  (req {req.date()}, Δ{abs((actual-req).days)}d)"
    else:  # static (y,x)
        img = da.values
        when = "static"

    img = np.asarray(img, dtype="float32")
    finite = img[np.isfinite(img)]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.85, label=args.var)
    ax.set_xticks([]); ax.set_yticks([])
    if finite.size:
        stats = (f"min {finite.min():.3g}  max {finite.max():.3g}  "
                 f"mean {finite.mean():.3g}  valid {finite.size/img.size:.0%}")
    else:
        stats = "all NaN/empty"
    ax.set_title(f"{args.aoi}: {args.var}\n{when}\n{stats}", fontsize=10)
    out = Path(args.out) / f"qc_{args.aoi}_{args.var}_{when.split()[0]}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  | {stats}")


if __name__ == "__main__":
    main()