#!/usr/bin/env python3
"""
Visualize which scenes the config will TRAIN on, and which pixels are included.

For a chosen AOI and date range, draws a contact sheet of every available
ECOSTRESS / Landsat scene, rendering the SST and annotating it the way the
training pipeline (data.py) would treat it:

  * a whole scene whose target day is REJECTED by Stage-A selection
    (select_max_cloud_cover_pct / select_min_*_frac) gets a thick RED BORDER
    and a red title;
  * individual pixels that have an observation but are REMOVED from the loss
    (valid mask + MUR filter + cloud gate + per-sensor loss_pixel_filters) are
    SHADED RED over the SST.

So: red border = scene dropped; red shading = pixels dropped. Anything left in
normal color is exactly what the model would be supervised on.

    python src/plot_selection.py --config configs/config.yaml --aoi hood_canal
    python src/plot_selection.py --config configs/config.yaml --aoi hood_canal \
        --start 2023-06-01 --end 2023-09-30 --sensor eco
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
from data import OceansrTileDataset, CHANNELS   # noqa: E402


def _present(arr, water, rng):
    """Range-gated finite over water -> the 'an observation exists here' mask."""
    fin = np.isfinite(arr)
    if rng is not None:
        fin &= (arr >= rng[0]) & (arr <= rng[1])
    return fin & water


def render_sheet(panels, sensor, out_path):
    n = len(panels)
    if n == 0:
        return 0
    ncol = min(6, n)
    nrow = math.ceil(n / ncol)
    finite = np.concatenate([p["sst"][np.isfinite(p["sst"])] for p in panels
                             if np.isfinite(p["sst"]).any()] or [np.array([np.nan])])
    vmin, vmax = (np.nanpercentile(finite, [2, 98]) if np.isfinite(finite).any()
                  else (None, None))
    cmap = plt.cm.viridis.copy(); cmap.set_bad(alpha=0.0)   # missing -> transparent

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.0 * nrow), squeeze=False)
    im = None
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n:
            ax.axis("off"); continue
        p = panels[k]
        im = ax.imshow(p["sst"], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        red = np.zeros((*p["sst"].shape, 4), "float32")          # red shade where dropped
        red[..., 0] = 1.0
        red[..., 3] = np.where(p["removed"], 0.55, 0.0)
        ax.imshow(red, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        if p["selected"]:
            ax.set_title(f"{p['date']}\nkept {p['kept_frac']:.0%}", fontsize=8)
        else:
            ax.set_title(f"{p['date']}\nSCENE REJECTED", fontsize=8, color="red")
            for sp in ax.spines.values():
                sp.set_color("red"); sp.set_linewidth(3)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=f"{sensor}_sst", location="right")
    fig.suptitle(f"{sensor.upper()} scenes  |  red border = scene dropped by selection  |  "
                 f"red shade = pixels dropped from loss", fontsize=11, y=0.99)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: cube start)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: cube end)")
    ap.add_argument("--sensor", default="both", choices=["eco", "lst", "both"])
    ap.add_argument("--mode", default="interior", choices=["interior", "last"])
    ap.add_argument("--out", default="results/selection")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = OceansrTileDataset(cfg, split="train", require_tiles=False)  # any AOI, incl. val
    zds = ds._zarr(args.aoi)
    times = pd.to_datetime(zds["time"].values)
    lo = 0 if not args.start else int(np.searchsorted(times.values, np.datetime64(args.start)))
    hi = len(times) - 1 if not args.end else int(
        np.searchsorted(times.values, np.datetime64(args.end), side="right")) - 1
    lo, hi = max(0, lo), min(len(times) - 1, hi)

    water = (zds["landmask"].values == 0)
    eco_idx = CHANNELS.index("eco_finite"); lst_idx = CHANNELS.index("lst_finite")
    eco_raw = zds["eco_sst"].isel(time=slice(lo, hi + 1)).values
    lst_raw = zds["lst_sst"].isel(time=slice(lo, hi + 1)).values
    eco_pres = _present(eco_raw, water[None], ds.sst_valid_range)
    lst_pres = _present(lst_raw, water[None], ds.sst_valid_range)

    want = []
    if args.sensor in ("eco", "both"):
        want += [("eco", lo + k) for k in range(hi - lo + 1) if eco_pres[k].any()]
    if args.sensor in ("lst", "both"):
        want += [("lst", lo + k) for k in range(hi - lo + 1) if lst_pres[k].any()]
    days = sorted({d for _, d in want})
    print(f"{args.aoi} {times[lo].date()}..{times[hi].date()} | "
          f"eco scenes={sum(s=='eco' for s,_ in want)} lst scenes={sum(s=='lst' for s,_ in want)}")

    # build the pipeline sample once per observed day (gives masks + selection)
    cache = {}
    for d in days:
        s = ds.build_sample(args.aoi, d, mode=args.mode, blank_target=False,
                            enforce_min=False, return_diagnostics=True)
        cache[d] = s

    Path(args.out).mkdir(parents=True, exist_ok=True)
    aoi = args.aoi
    n_water = max(int(water.sum()), 1)
    written = []
    for sensor, presvol, sidx, sstvol in (("eco", eco_pres, eco_idx, eco_raw),
                                          ("lst", lst_pres, lst_idx, lst_raw)):
        if args.sensor not in (sensor, "both"):
            continue
        panels = []
        for k in range(hi - lo + 1):
            d = lo + k
            if not presvol[k].any():
                continue
            s = cache[d]
            kept = s[f"{sensor}_mask"].numpy() > 0.5            # supervised pixels
            pres = presvol[k]                                   # obs exists (water)
            sst = np.where(pres, sstvol[k], np.nan)
            panels.append({
                "date": times[d].strftime("%Y-%m-%d"),
                "sst": sst,
                "removed": pres & ~kept,                        # has obs but dropped
                "selected": bool(s.get("selection_ok", True)),
                "kept_frac": float(kept.sum()) / n_water,
            })
        out = Path(args.out) / f"{aoi}_{sensor}_selection.png"
        if render_sheet(panels, sensor, out):
            written.append(str(out)); print(f"  wrote {out} ({len(panels)} scenes)")
    if not written:
        print("no scenes in range for the chosen sensor(s).")


if __name__ == "__main__":
    main()