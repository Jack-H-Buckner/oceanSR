#!/usr/bin/env python3
"""
Plot the OCEANSR AOIs on a national overview and regional zoom maps, colored by
region. Reads the same configs/config.yaml the pipeline uses.

    python src/plot_aois.py --config configs/config.yaml

Outputs (to results/):
    aoi_overview.png     # national + 4 regional panels in one figure
    aoi_national.png
    aoi_<panel>.png      # one per regional panel

Uses matplotlib. If `cartopy` is installed AND Natural Earth basemap data can be
fetched (first run needs internet; cached after), coastlines/states are drawn.
Otherwise the AOIs still plot on a lon/lat grid so the script always works.
PNG export is pure matplotlib -- no headless browser needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# --- optional cartopy basemap ------------------------------------------------
USE_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    # Probe: force one Natural Earth fetch so we fail fast (and offline) here,
    # not lazily at savefig time.
    shapereader.natural_earth(resolution="50m", category="cultural",
                              name="admin_1_states_provinces_lines")
    PC = ccrs.PlateCarree()
    USE_CARTOPY = True
except Exception as exc:  # cartopy missing or basemap unreachable
    print(f"[info] cartopy basemap unavailable ({type(exc).__name__}); "
          f"plotting AOIs without coastlines.")
    PC = None

REGION_COLORS = {
    "PNW Estuaries": "#1f77b4",
    "Salish Sea":    "#17becf",
    "SE Alaska":     "#9467bd",
    "Chesapeake":    "#2ca02c",
    "Carolinas":     "#ff7f0e",
    "New England":   "#d62728",
}

PANELS = {
    "pacific_northwest": dict(
        title="Pacific Northwest (estuaries + Salish Sea)",
        extent=[-125.4, -121.8, 42.9, 49.0],
        regions=["PNW Estuaries", "Salish Sea"]),
    "se_alaska": dict(
        title="Southeast Alaska fjords",
        extent=[-137.6, -130.3, 54.7, 59.6],
        regions=["SE Alaska"]),
    "chesapeake_carolinas": dict(
        title="Chesapeake Bay & Carolina sounds",
        extent=[-80.5, -74.7, 31.9, 39.8],
        regions=["Chesapeake", "Carolinas"]),
    "new_england": dict(
        title="New England coast",
        extent=[-71.9, -67.7, 41.0, 44.8],
        regions=["New England"]),
}

NATIONAL_EXTENT = [-140.0, -66.0, 24.0, 60.0]   # [W, E, S, N]


def load_aois(config_path: str) -> list[dict]:
    return yaml.safe_load(open(config_path))["aois"]


def center(a):
    w, s, e, n = a["bbox"]
    return (w + e) / 2.0, (s + n) / 2.0


def color_for(a):
    return REGION_COLORS.get(a.get("region", ""), "#555555")


def make_ax(fig, spec, extent):
    """Create an axis for the given [W,E,S,N] extent, with or without cartopy."""
    w, e, s, n = extent
    if USE_CARTOPY:
        ax = fig.add_subplot(spec, projection=PC)
        ax.set_extent(extent, crs=PC)
        ax.add_feature(cfeature.LAND, facecolor="#eee9df")
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3ef")
        ax.add_feature(cfeature.LAKES, facecolor="#cfe3ef")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#999999")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#cccccc")
        gl.top_labels = gl.right_labels = False
    else:
        ax = fig.add_subplot(spec)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect(1.0 / np.cos(np.deg2rad((s + n) / 2.0)))  # ~equal-area look
        ax.grid(True, linewidth=0.3, color="#cccccc")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    return ax


def _kw():
    return dict(transform=PC) if USE_CARTOPY else {}


def draw_national(ax, aois):
    for a in aois:
        lon, lat = center(a)
        ax.plot(lon, lat, "o", ms=5, color=color_for(a),
                markeredgecolor="k", markeredgewidth=0.4, zorder=5, **_kw())
    ax.set_title(f"OCEANSR AOIs (n={len(aois)})", fontsize=12, fontweight="bold")
    regions = [r for r in REGION_COLORS if any(a.get("region") == r for a in aois)]
    ax.legend(handles=[Patch(color=REGION_COLORS[r], label=r) for r in regions],
              loc="lower left", fontsize=7, framealpha=0.9)


def draw_panel(ax, aois, panel):
    for a in [x for x in aois if x.get("region") in panel["regions"]]:
        w, s, e, n = a["bbox"]
        ax.add_patch(Rectangle((w, s), e - w, n - s, facecolor=color_for(a),
                               edgecolor=color_for(a), alpha=0.4, linewidth=1.2,
                               zorder=5, **_kw()))
        ax.text((w + e) / 2, n, a["id"], fontsize=6, ha="center", va="bottom",
                zorder=6, **_kw())
    ax.set_title(panel["title"], fontsize=10)


def main():
    ap = argparse.ArgumentParser(description="Plot OCEANSR AOIs.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    aois = load_aois(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Combined overview: national on top, 4 regional panels below.
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], hspace=0.2, wspace=0.12)
    draw_national(make_ax(fig, gs[0, :], NATIONAL_EXTENT), aois)
    for i, panel in enumerate(PANELS.values()):
        draw_panel(make_ax(fig, gs[1 + i // 2, i % 2], panel["extent"]), aois, panel)
    fig.suptitle("OCEANSR Areas of Interest", fontsize=15, fontweight="bold", y=0.995)
    fig.savefig(out / "aoi_overview.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out/'aoi_overview.png'}")

    # Individual figures.
    f = plt.figure(figsize=(11, 7))
    draw_national(make_ax(f, 111, NATIONAL_EXTENT), aois)
    f.savefig(out / "aoi_national.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out/'aoi_national.png'}")
    for key, panel in PANELS.items():
        f = plt.figure(figsize=(7, 7))
        draw_panel(make_ax(f, 111, panel["extent"]), aois, panel)
        f.savefig(out / f"aoi_{key}.png", dpi=150, bbox_inches="tight")
        print(f"wrote {out/('aoi_'+key+'.png')}")


if __name__ == "__main__":
    main()