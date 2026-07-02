#!/usr/bin/env python3
"""
Inspect an ECOSTRESS (or Landsat) aligned overpass file to work out the correct
masking logic empirically -- which value of the water/cloud/QC layers actually
corresponds to a valid WATER retrieval.

It plots every layer and, crucially, cross-tabulates the LST/SST values against
each mask class so you can SEE which class is sea-surface temperature.

    python src/inspect_ecostress.py --aoi hood_canal --date 2023-07-15
    python src/inspect_ecostress.py --aoi hood_canal --aligned-dir data/ECOSTRESS/aligned
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DT = re.compile(r"(\d{8}T\d{6})")
CONT = {"sst", "lst", "quality"}     # show with a continuous colormap


def pick_file(files, date):
    if date:
        tgt = datetime.strptime(date, "%Y-%m-%d")
        return min(files, key=lambda f: abs(
            (datetime.strptime(_DT.search(f.name).group(1), "%Y%m%dT%H%M%S") - tgt).days)
            if _DT.search(f.name) else 1e9)
    # else the file with the most finite sst/lst
    def n_finite(f):
        d = xr.open_dataset(f)
        v = d["sst"] if "sst" in d else list(d.data_vars.values())[0]
        n = int(np.isfinite(v.values).sum()); d.close(); return n
    return max(files, key=n_finite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--aligned-dir", default="data/ECOSTRESS/aligned")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    files = sorted((Path(args.aligned_dir) / args.aoi).glob(f"{args.aoi}_*.nc"))
    if not files:
        raise SystemExit(f"no aligned files in {Path(args.aligned_dir)/args.aoi}")
    f = pick_file(files, args.date)
    print(f"inspecting {f.name}")
    ds = xr.open_dataset(f)
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    temp_name = "sst" if "sst" in ds else ("lst" if "lst" in ds else None)
    temp = ds[temp_name].values.astype("float32") if temp_name else None
    layers = [v for v in ("sst", "lst", "cloud", "water", "quality", "valid") if v in ds]

    # ---- value tables + cross-tab against LST ---------------------------- #
    print("\nlayer value distributions:")
    for m in ("cloud", "water", "quality", "valid"):
        if m in ds:
            vals, cnts = np.unique(ds[m].values[np.isfinite(ds[m].values)], return_counts=True)
            print(f"  {m:8s}: " + ", ".join(f"{v:g}={c}" for v, c in zip(vals, cnts)))

    if temp is not None:
        print(f"\n{temp_name.upper()} (K) by mask class  [water ~270-295K]:")
        for m in ("water", "cloud", "quality"):
            if m not in ds:
                continue
            mv = ds[m].values
            for v in np.unique(mv[np.isfinite(mv)]):
                sel = (mv == v) & np.isfinite(temp)
                if sel.any():
                    t = temp[sel]
                    print(f"  {m}=={v:g}: {temp_name} mean {t.mean():6.1f}  "
                          f"min {t.min():6.1f}  max {t.max():6.1f}  n={sel.sum()}")

    # ---- QC mandatory-QA decode (bits 0-1; V2 has NO cloud bit) ---------- #
    if "quality" in ds:
        q = ds["quality"].values
        mqa = np.full(q.shape, -1, dtype="int64")
        fin = np.isfinite(q)
        mqa[fin] = q[fin].astype("int64") & 0b11
        labels = {0: "produced/best", 1: "produced/degraded",
                  2: "(unset)", 3: "not produced", -1: "nodata"}
        print("\nQC mandatory-QA (bits 0-1)  [00/01 = usable]:")
        for v in np.unique(mqa):
            sel = mqa == v
            line = f"  {v:2d} {labels.get(int(v), ''):18s} n={int(sel.sum())}"
            if temp is not None:
                t = temp[sel & np.isfinite(temp)]
                if t.size:
                    line += f"   {temp_name} mean {t.mean():.1f}K"
            print(line)

    # ---- panels --------------------------------------------------------- #
    items = [(name, ds[name].values.astype("float32"),
              ("viridis" if name in CONT else "tab10"), (None, None))
             for name in layers]
    if "quality" in ds:   # add a DECODED QC panel (bits 0-1), more readable than raw
        items.append(("QC quality grade", mqa.astype("float32"), "viridis", (-0.5, 3.5)))
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4), squeeze=False)
    for ax, (title, img, cmap, (lo, hi)) in zip(axes[0], items):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        if title == "QC quality grade":      # label the QC classes
            cb.set_ticks([0, 1, 2, 3])
            cb.set_ticklabels(["0 best", "1 degraded", "2 unset", "3 not prod"])
    fig.suptitle(f"{args.aoi}  {f.name}", fontsize=11)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"inspect_{args.aoi}_{f.stem}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()