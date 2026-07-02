#!/usr/bin/env python3
"""
Diagnose what feeds the derived high-res validity mask (valid_from_sst path).
Answers: are non-observation gaps NaN or a fill value? is SST in Kelvin? how
many days actually carry an overpass, and what fraction of water ends up flagged?

    python src/inspect_mask.py --config configs/config.yaml --aoi hood_canal
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg.get("project", {}).get("root", "."))
    zdir = root / cfg["paths"]["data"] / cfg["paths"]["training"]
    tcfg = cfg.get("train", {})
    sensors = [s.lower() for s in tcfg.get("sensors", ["eco", "lst"])]
    mur_thr = float(cfg.get("assembler", {}).get("mur_cloud_threshold_k", 5.0))

    ds = xr.open_zarr(zdir / f"{args.aoi}.zarr")
    nt = ds.sizes["time"]
    water = (ds["landmask"].values == 0)
    nwater = int(water.sum())
    mur = ds["mur_sst"].values if "mur_sst" in ds else None
    print(f"{args.aoi}: T={nt}  water px/day={nwater}  sensors={sensors}  mur_thr={mur_thr}K")

    for s in sensors:
        v = f"{s}_sst"
        if v not in ds:
            print(f"  [{s}] {v} MISSING from cube -> _fetch returns ZEROS "
                  f"-> isfinite=True everywhere -> mask saturates. (root cause)")
            continue
        a = ds[v].values.astype("float32")          # (T,H,W)
        wmask3 = np.broadcast_to(water, a.shape)
        wvals = a[wmask3]
        finite = np.isfinite(wvals)
        nan_frac = 1.0 - finite.mean()
        fin = wvals[finite]
        zero_frac = (np.abs(fin) < 1e-6).mean() if fin.size else 0.0
        rng = (float(np.nanmin(fin)), float(np.nanmax(fin))) if fin.size else (np.nan, np.nan)
        units = ("Kelvin" if rng[1] > 200 else "Celsius?" if rng[1] < 60 else "??")
        # per-day finite-over-water fraction -> how many days look like overpasses
        per_day = np.array([np.isfinite(a[t][water]).mean() for t in range(nt)])
        overpass = int((per_day > 0.01).sum())
        # what the CURRENT mask would flag (finite & water [& mur filter])
        finmask = np.isfinite(a) & water[None]
        if mur is not None and mur_thr > 0:
            finmask &= ~((mur - a) > mur_thr)
        flagged_frac = finmask.sum() / (nt * nwater)

        print(f"  [{s}] finite-over-water: {finite.mean():.1%}  (NaN gaps: {nan_frac:.1%})")
        print(f"       finite values: min={rng[0]:.2f} max={rng[1]:.2f}  -> {units}; "
              f"exact-zero among finite: {zero_frac:.1%}")
        print(f"       days with any finite water (overpass-like): {overpass}/{nt}")
        print(f"       => CURRENT mask flags {flagged_frac:.1%} of all water-pixel-days "
              f"as 'valid' (want roughly overpass_days/T).")


if __name__ == "__main__":
    main()