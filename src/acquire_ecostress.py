#!/usr/bin/env python3
"""
OCEANSR — ECOSTRESS L2T LSTE (V3) acquisition.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; ECOSTRESS-specific settings come from `sources.ecostress`.
For each AOI it searches ECO_L2T_LSTE by bbox + dates, then for every overpass
*streams a windowed read* of the SST + cloud/water/QC COGs (HTTP range requests
via earthaccess.open -- no full-tile download to disk), reprojects the AOI window
onto the AOI grid, clips to the AOI polygon, and writes one aligned file per
overpass into data/ECOSTRESS/aligned/. A later stage bins these to a daily cube.

Because only the AOI window is read and only the small aligned outputs are kept,
there is no multi-GB raw tile cache.

Usage (run from the OCEANSR project root):
    python src/acquire_ecostress.py --config configs/config.yaml
    python src/acquire_ecostress.py --config configs/config.yaml --dry-run
    python src/acquire_ecostress.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_ecostress.py --config configs/config.yaml --list-layers
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from datetime import datetime
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

log = logging.getLogger("acquire_ecostress")
SOURCE = "ecostress"

_DT_RE = re.compile(r"(\d{8}T\d{6})")  # ..._20230715T210043_....tif


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_effective(cfg: dict, source: str) -> dict:
    """Merge shared blocks with per-source overrides and resolve project paths."""
    if source not in cfg.get("sources", {}):
        raise SystemExit(f"Source '{source}' not found under 'sources' in config.")
    src_cfg = cfg["sources"][source]

    # Shared time/grid, with optional per-source overrides.
    time_cfg = {**cfg.get("time", {}), **src_cfg.get("time", {})}
    grid_cfg = {**cfg.get("grid", {}), **src_cfg.get("grid", {})}

    # Resolve the OCEANSR data layout: data/<SOURCE_DIR>/{raw,aligned}.
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    src_dir = root / paths["data"] / paths[source]

    return {
        "earthdata": cfg["earthdata"],
        "aois": cfg["aois"],
        "time": time_cfg,
        "grid": grid_cfg,
        "ds": src_cfg,
        "out_dir": src_dir / "aligned",   # streamed reads -> no raw tile cache
        "fmt": src_cfg.get("output_format", "netcdf"),
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry / grid helpers
# --------------------------------------------------------------------------- #
def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def aoi_geometry_4326(aoi: dict):
    if aoi.get("geometry"):
        with open(aoi["geometry"]) as f:
            gj = json.load(f)
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
    geom_proj = shp_transform(fwd, geom_4326)
    if buffer_m and buffer_m > 0:
        geom_proj = geom_proj.buffer(buffer_m)
    return geom_proj


def search_bbox_4326(geom_proj, target_crs: str):
    inv = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True).transform
    minx, miny, maxx, maxy = shp_transform(inv, geom_proj).bounds
    return (minx, miny, maxx, maxy)  # (W, S, E, N)


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
# Earthdata search / download
# --------------------------------------------------------------------------- #
def login(strategy: str):
    log.info("Authenticating with Earthdata (strategy=%s)", strategy)
    earthaccess.login(strategy=strategy)


def search_granules(ds_cfg: dict, bbox, start: str, end: str):
    results = earthaccess.search_data(
        short_name=ds_cfg["short_name"],
        version=ds_cfg["version"],
        temporal=(start, end),
        bounding_box=tuple(bbox),
    )
    log.info("  found %d granule(s)", len(results))
    return results


def granule_name(granule) -> str:
    try:
        return granule.data_links()[0].split("/")[-1]
    except Exception:
        return "<granule>"


def filter_links_for_granule(granule, layers: dict) -> dict:
    """{role: url} for the COG assets we want from one granule."""
    links = granule.data_links()
    out = {}
    for role, suffix in layers.items():
        tail = f"_{suffix}.tif"
        match = next((u for u in links if u.endswith(tail)), None)
        if match:
            out[role] = match
        elif role == "sst":
            log.warning("    SST asset not found in %s", granule_name(granule))
    return out


def parse_acq_time(filename: str) -> Optional[datetime]:
    m = _DT_RE.search(filename)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None


# --------------------------------------------------------------------------- #
# Per-granule processing
# --------------------------------------------------------------------------- #
def read_window_reproject(fobj, geom_target, target_crs, transform, width,
                          height, resampling):
    """Open a remote COG, read ONLY the AOI window, reproject to the AOI grid.

    `fobj` is an fsspec file object from earthaccess.open(); rioxarray reads it
    lazily, so clip_box triggers HTTP range reads of just the windowed blocks
    rather than pulling the whole 110 km tile.
    """
    da = rioxarray.open_rasterio(fobj, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    cog_crs = da.rio.crs
    # AOI bounds expressed in the COG's CRS, padded a few pixels for clean edges.
    to_cog = Transformer.from_crs(target_crs, cog_crs, always_xy=True).transform
    minx, miny, maxx, maxy = shp_transform(to_cog, geom_target).bounds
    pad = 600.0  # ~6 pixels of slack (COG units are metres)
    da = da.rio.clip_box(minx - pad, miny - pad, maxx + pad, maxy + pad, crs=cog_crs)
    return da.rio.reproject(
        dst_crs=target_crs, shape=(height, width), transform=transform,
        resampling=resampling, nodata=np.nan,
    )


def process_granule(role_to_file, ds_cfg, grid_cfg, target_crs, transform,
                    width, height, geom_proj, aoi_id, acq_time) -> Optional[xr.Dataset]:
    categorical = set(ds_cfg.get("categorical", []))
    rs_cont = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    rs_cat = Resampling[grid_cfg.get("resampling_categorical", "nearest")]

    data_vars = {}
    for role, fobj in role_to_file.items():
        resampling = rs_cat if ds_cfg["layers"][role] in categorical else rs_cont
        try:
            da = read_window_reproject(fobj, geom_proj, target_crs, transform,
                                       width, height, resampling)
            da = da.rio.clip([geom_proj], target_crs, drop=False)
        except Exception as exc:  # window outside this tile, etc.
            log.warning("    skipping layer %s (%s)", role, exc)
            continue
        data_vars[role] = da

    if "sst" not in data_vars and "lst" not in data_vars:
        log.warning("    no SST/LST after processing; dropping granule")
        return None

    ds = xr.Dataset(data_vars)

    if grid_cfg.get("to_celsius", False):
        for v in ("sst", "lst"):
            if v in ds:
                ds[v] = ds[v] - 273.15
                ds[v].attrs["units"] = "degC"
    else:
        for v in ("sst", "lst"):
            if v in ds:
                ds[v].attrs["units"] = "K"

    if {"sst", "water", "cloud"} <= set(ds.data_vars):
        valid = np.isfinite(ds["sst"]) & (ds["water"] > 0) & ~(ds["cloud"] > 0)
        ds["valid"] = valid.astype("uint8")
        ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    if acq_time is not None:
        ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds.attrs.update(aoi_id=aoi_id, source="ECOSTRESS ECO_L2T_LSTE v003",
                    processing="reprojected+clipped to AOI grid")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(ds["time"].values[0]).strftime("%Y%m%dT%H%M%S") \
        if "time" in ds.coords else "unknown"
    stem = f"{aoi_id}_{t}"
    if fmt == "netcdf":
        path = out_dir / f"{stem}.nc"
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(path, encoding=enc)
    elif fmt == "geotiff":
        path = out_dir / stem
        path.mkdir(exist_ok=True)
        for v in ds.data_vars:
            da = ds[v].isel(time=0) if "time" in ds[v].dims else ds[v]
            da.rio.to_raster(path / f"{v}.tif")
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff: dict, only_aoi, dry_run, list_layers):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root = eff["out_dir"]
    fmt, overwrite = eff["fmt"], eff["overwrite"]

    login(eff["earthdata"]["auth_strategy"])
    layers = dict(ds_cfg["layers"])

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
        bbox = search_bbox_4326(geom_proj, target_crs)
        transform, width, height = build_target_grid(
            geom_proj, grid_cfg["resolution_m"], grid_cfg.get("snap_origin", True))
        log.info("  target CRS=%s grid=%dx%d @ %sm", target_crs, width, height,
                 grid_cfg["resolution_m"])

        granules = search_granules(ds_cfg, bbox, eff["time"]["start_date"],
                                   eff["time"]["end_date"])
        if not granules:
            continue

        if list_layers:
            links = granules[0].data_links()
            tails = sorted({u.rsplit("_", 1)[-1] for u in links if u.endswith(".tif")})
            log.info("  available COG suffixes: %s", tails)
            continue
        if dry_run:
            log.info("  [dry-run] would process %d granule(s)", len(granules))
            continue

        aoi_out = out_root / aoi_id

        for gi, granule in enumerate(granules, 1):
            role_to_url = filter_links_for_granule(granule, layers)
            if not role_to_url:
                continue
            t = parse_acq_time(granule_name(granule))
            tstr = t.strftime("%Y%m%dT%H%M%S") if t else f"g{gi}"
            if not overwrite and (aoi_out / f"{aoi_id}_{tstr}.nc").exists():
                log.info("  [%d/%d] %s already processed, skipping", gi, len(granules), tstr)
                continue

            log.info("  [%d/%d] streaming %d layer(s) for %s",
                     gi, len(granules), len(role_to_url), tstr)
            try:
                fobjs = earthaccess.open(list(role_to_url.values()))
            except Exception as exc:
                log.warning("    open failed (%s); skipping", exc)
                continue
            role_to_file = dict(zip(role_to_url.keys(), fobjs))

            ds = process_granule(role_to_file, ds_cfg, grid_cfg, target_crs,
                                 transform, width, height, geom_proj, aoi_id, t)
            if ds is None:
                continue
            log.info("      wrote %s", write_output(ds, aoi_out, aoi_id, fmt))

    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR ECOSTRESS V3 SST acquisition.")
    ap.add_argument("--config", required=True, help="Path to configs/config.yaml.")
    ap.add_argument("--source", default=SOURCE, help="Source key under 'sources' (default: ecostress).")
    ap.add_argument("--aoi", help="Process only this AOI id.")
    ap.add_argument("--dry-run", action="store_true", help="Search only; no download.")
    ap.add_argument("--list-layers", action="store_true",
                    help="Print available COG suffixes for the first granule and exit.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Make GDAL/curl efficient for remote COG range reads.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    cfg = load_config(args.config)
    eff = build_effective(cfg, args.source)
    run(eff, args.aoi, args.dry_run, args.list_layers)


if __name__ == "__main__":
    main()