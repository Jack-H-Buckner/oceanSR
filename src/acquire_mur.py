#!/usr/bin/env python3
"""
OCEANSR -- MUR L4 SST backbone acquisition.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; MUR settings come from `sources.mur`. For each AOI and each
day it streams the daily GHRSST MUR L4 granule from PODAAC (earthaccess.open),
subsets `analysed_sst` to the AOI lat/lon window (HDF5 range reads -- the global
1 km file is never fully downloaded), upsamples onto the AOI grid (identical to
the ECOSTRESS/Landsat grid), and writes one aligned NetCDF per day.

MUR is a gap-free L4 analysis, so it has no cloud mask; `valid` = finite SST
(i.e. water). It is the always-present backbone the model fills high-res detail
onto. A later stage bins these to the daily datacube.

Usage (run from the OCEANSR project root, Earthdata auth via ~/.netrc):
    python src/acquire_mur.py --config configs/config.yaml
    python src/acquire_mur.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_mur.py --config configs/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import earthaccess
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

log = logging.getLogger("acquire_mur")
SOURCE = "mur"


# --------------------------------------------------------------------------- #
# Config plumbing (mirrors the other stages)
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_effective(cfg: dict, source: str) -> dict:
    if source not in cfg.get("sources", {}):
        raise SystemExit(f"Source '{source}' not found under 'sources' in config.")
    src_cfg = cfg["sources"][source]
    time_cfg = {**cfg.get("time", {}), **src_cfg.get("time", {})}
    grid_cfg = {**cfg.get("grid", {}), **src_cfg.get("grid", {})}
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    src_dir = root / paths["data"] / paths[source]
    return {
        "earthdata": cfg["earthdata"],
        "aois": cfg["aois"],
        "time": time_cfg,
        "grid": grid_cfg,
        "ds": src_cfg,
        "out_dir": src_dir / "aligned",
        "fmt": src_cfg.get("output_format", "netcdf"),
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry / grid helpers (same conventions as the other stages)
# --------------------------------------------------------------------------- #
def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def aoi_geometry_4326(aoi: dict):
    if aoi.get("geometry"):
        import json
        gj = json.load(open(aoi["geometry"]))
        if gj.get("type") == "FeatureCollection":
            return unary_union([shape(f["geometry"]) for f in gj["features"]])
        if gj.get("type") == "Feature":
            return shape(gj["geometry"])
        return shape(gj)
    if aoi.get("bbox"):
        w, s, e, n = aoi["bbox"]
        return box(w, s, e, n)
    raise ValueError(f"AOI '{aoi.get('id')}' needs either 'bbox' or 'geometry'.")


def resolve_target_crs(geom_4326, grid_cfg: dict) -> str:
    tc = str(grid_cfg.get("target_crs", "auto")).lower()
    if tc == "auto":
        c = geom_4326.centroid
        return utm_epsg_from_lonlat(c.x, c.y)
    return grid_cfg["target_crs"]


def buffered_geom_in_crs(geom_4326, target_crs: str, buffer_m: float):
    fwd = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
    g = shp_transform(fwd, geom_4326)
    return g.buffer(buffer_m) if buffer_m and buffer_m > 0 else g


def latlon_bounds(geom_proj, target_crs: str):
    inv = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True).transform
    return shp_transform(inv, geom_proj).bounds  # (W, S, E, N)


def build_target_grid(geom_proj, resolution_m: float, snap: bool):
    minx, miny, maxx, maxy = geom_proj.bounds
    r = float(resolution_m)
    if snap:
        minx = math.floor(minx / r) * r
        miny = math.floor(miny / r) * r
        maxx = math.ceil(maxx / r) * r
        maxy = math.ceil(maxy / r) * r
    width = int(round((maxx - minx) / r))
    height = int(round((maxy - miny) / r))
    return from_origin(minx, maxy, r, r), width, height


# --------------------------------------------------------------------------- #
# MUR per-granule processing
# --------------------------------------------------------------------------- #
def subset_and_reproject(fobj, variable, bbox_ll, pad, target_crs, transform,
                         width, height, geom_proj, grid_cfg) -> tuple[xr.DataArray, pd.Timestamp]:
    """Open one daily MUR granule lazily, subset to the AOI, upsample to grid."""
    w, s, e, n = bbox_ll
    ds = xr.open_dataset(fobj, engine="h5netcdf", mask_and_scale=True)
    da = ds[variable].isel(time=0)
    da = da.sel(lat=slice(s - pad, n + pad), lon=slice(w - pad, e + pad)).load()

    t = pd.Timestamp(ds["time"].values[0]).tz_localize(None)

    if grid_cfg.get("to_celsius", False):
        da = da - 273.15

    # Standard orientation + CRS, then reproject (bilinear upsample 1 km -> grid).
    da = da.rename({"lon": "x", "lat": "y"}).sortby("y", ascending=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    rs = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    out = da.rio.reproject(dst_crs=target_crs, shape=(height, width),
                           transform=transform, resampling=rs, nodata=np.nan)
    out = out.rio.clip([geom_proj], target_crs, drop=False)
    return out, t


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.Timestamp(ds["time"].values[0]).strftime("%Y%m%d")
    stem = f"{aoi_id}_{d}"
    if fmt == "netcdf":
        path = out_dir / f"{stem}.nc"
        ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    elif fmt == "geotiff":
        path = out_dir / stem
        path.mkdir(exist_ok=True)
        for v in ds.data_vars:
            ds[v].isel(time=0).rio.to_raster(path / f"{v}.tif")
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, only_aoi, dry_run):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    variable = ds_cfg.get("variable", "analysed_sst")
    pad = float(ds_cfg.get("pad_deg", 0.05))
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    log.info("Authenticating with Earthdata (strategy=%s)", eff["earthdata"]["auth_strategy"])
    earthaccess.login(strategy=eff["earthdata"]["auth_strategy"])

    aois = eff["aois"]
    if only_aoi:
        aois = [a for a in aois if a["id"] == only_aoi]
        if not aois:
            raise SystemExit(f"AOI '{only_aoi}' not found in config.")

    for aoi in aois:
        aoi_id = aoi["id"]
        log.info("=== AOI: %s (%s) ===", aoi_id, aoi.get("name", ""))
        geom_4326 = aoi_geometry_4326(aoi)
        target_crs = resolve_target_crs(geom_4326, grid_cfg)
        geom_proj = buffered_geom_in_crs(geom_4326, target_crs, aoi.get("buffer_m", 0))
        bbox_ll = latlon_bounds(geom_proj, target_crs)
        transform, width, height = build_target_grid(
            geom_proj, grid_cfg["resolution_m"], grid_cfg.get("snap_origin", True))
        log.info("  target CRS=%s grid=%dx%d @ %sm", target_crs, width, height,
                 grid_cfg["resolution_m"])

        granules = earthaccess.search_data(
            short_name=ds_cfg["short_name"], temporal=(start, end),
            bounding_box=tuple(bbox_ll))
        log.info("  %d daily MUR granule(s)", len(granules))
        if not granules or dry_run:
            if dry_run:
                log.info("  [dry-run] would process %d day(s)", len(granules))
            continue

        aoi_out = out_root / aoi_id
        for gi, granule in enumerate(granules, 1):
            try:
                fobj = earthaccess.open([granule])[0]
                da, t = subset_and_reproject(fobj, variable, bbox_ll, pad, target_crs,
                                             transform, width, height, geom_proj, grid_cfg)
            except Exception as exc:
                log.warning("    [%d/%d] skipping (%s)", gi, len(granules), exc)
                continue

            dstr = t.strftime("%Y%m%d")
            if not overwrite and (aoi_out / f"{aoi_id}_{dstr}.nc").exists():
                log.info("  [%d/%d] %s already processed, skipping", gi, len(granules), dstr)
                continue

            ds = xr.Dataset({"sst": da})
            ds["sst"].attrs["units"] = "degC" if grid_cfg.get("to_celsius", False) else "K"
            ds["valid"] = np.isfinite(ds["sst"]).astype("uint8")
            ds["valid"].attrs["long_name"] = "finite MUR SST (water)"
            ds = ds.expand_dims(time=[t])
            ds.attrs.update(aoi_id=aoi_id, source=f"GHRSST {ds_cfg['short_name']}",
                            processing="subset + bilinear upsample to AOI grid")
            log.info("  [%d/%d] wrote %s", gi, len(granules),
                     write_output(ds, aoi_out, aoi_id, fmt))
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR MUR L4 SST backbone acquisition.")
    ap.add_argument("--config", required=True, help="Path to configs/config.yaml.")
    ap.add_argument("--aoi", help="Process only this AOI id.")
    ap.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    eff = build_effective(cfg, SOURCE)
    run(eff, args.aoi, args.dry_run)


if __name__ == "__main__":
    main()