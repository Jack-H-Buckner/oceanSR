#!/usr/bin/env python3
"""
OCEANSR -- bathymetry static covariate (GMRT GridServer).

Reads the common project config (configs/config.yaml): AOIs and grid are shared;
settings come from `sources.bathymetry`. For each AOI it requests a bathymetry
GeoTIFF for the AOI bounding box from the GMRT GridServer (best-available
resolution, GEBCO-filled), reprojects onto the AOI grid (identical to the SST
stages), and writes ONE static NetCDF per AOI (no time dimension):

    data/BATHYMETRY/aligned/<aoi_id>/<aoi_id>.nc

Variables: `elevation` (m, negative below sea level) and `depth` (m, = -elevation
over water, 0 on land). Only `requests` is needed beyond the SST-stage deps.

Usage (run from the project root):
    python src/acquire_bathymetry.py --config configs/config.yaml
    python src/acquire_bathymetry.py --config configs/config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import requests
import xarray as xr
import yaml

import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

log = logging.getLogger("acquire_bathymetry")
SOURCE = "bathymetry"
GMRT_URL = "https://www.gmrt.org/services/GridServer"


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_effective(cfg: dict, source: str) -> dict:
    if source not in cfg.get("sources", {}):
        raise SystemExit(f"Source '{source}' not found under 'sources' in config.")
    src_cfg = cfg["sources"][source]
    grid_cfg = {**cfg.get("grid", {}), **src_cfg.get("grid", {})}
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    return {
        "aois": cfg["aois"],
        "grid": grid_cfg,
        "ds": src_cfg,
        "out_dir": root / paths["data"] / paths[source] / "aligned",
        "tmp_dir": root / paths["data"] / paths[source] / "_tmp",
        "fmt": src_cfg.get("output_format", "netcdf"),
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry / grid helpers
# --------------------------------------------------------------------------- #
def utm_epsg_from_lonlat(lon, lat) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def aoi_geometry_4326(aoi):
    if aoi.get("geometry"):
        import json
        gj = json.load(open(aoi["geometry"]))
        if gj.get("type") == "FeatureCollection":
            return unary_union([shape(f["geometry"]) for f in gj["features"]])
        if gj.get("type") == "Feature":
            return shape(gj["geometry"])
        return shape(gj)
    w, s, e, n = aoi["bbox"]
    return box(w, s, e, n)


def resolve_target_crs(geom_4326, grid_cfg) -> str:
    tc = str(grid_cfg.get("target_crs", "auto")).lower()
    if tc == "auto":
        c = geom_4326.centroid
        return utm_epsg_from_lonlat(c.x, c.y)
    return grid_cfg["target_crs"]


def buffered_geom_in_crs(geom_4326, target_crs, buffer_m):
    fwd = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
    g = shp_transform(fwd, geom_4326)
    return g.buffer(buffer_m) if buffer_m and buffer_m > 0 else g


def latlon_bounds(geom_proj, target_crs):
    inv = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True).transform
    return shp_transform(inv, geom_proj).bounds  # (W, S, E, N)


def build_target_grid(geom_proj, resolution_m, snap):
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
# GMRT fetch
# --------------------------------------------------------------------------- #
def fetch_gmrt(bbox_ll, pad, layer, resolution, tmp_path: Path) -> Path:
    w, s, e, n = bbox_ll
    params = {
        "west": w - pad, "east": e + pad, "south": s - pad, "north": n + pad,
        "format": "geotiff", "layer": layer, "resolution": resolution,
    }
    r = requests.get(GMRT_URL, params=params, timeout=180)
    r.raise_for_status()
    if not r.content[:2] in (b"II", b"MM"):  # TIFF magic (little/big endian)
        raise RuntimeError(f"GMRT did not return a GeoTIFF (got {r.content[:80]!r})")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(r.content)
    return tmp_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def write_output(ds, out_dir, aoi_id, fmt) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        path = out_dir / f"{aoi_id}.nc"
        ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    elif fmt == "geotiff":
        path = out_dir / aoi_id
        path.mkdir(exist_ok=True)
        for v in ds.data_vars:
            ds[v].rio.to_raster(path / f"{v}.tif")
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    return path


def run(eff, only_aoi, dry_run):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, tmp_dir, fmt, overwrite = eff["out_dir"], eff["tmp_dir"], eff["fmt"], eff["overwrite"]
    pad = float(ds_cfg.get("pad_deg", 0.02))
    layer = ds_cfg.get("layer", "topo")
    resolution = ds_cfg.get("resolution", "max")

    aois = eff["aois"]
    if only_aoi:
        aois = [a for a in aois if a["id"] == only_aoi]
        if not aois:
            raise SystemExit(f"AOI '{only_aoi}' not found in config.")

    for aoi in aois:
        aoi_id = aoi["id"]
        geom_4326 = aoi_geometry_4326(aoi)
        target_crs = resolve_target_crs(geom_4326, grid_cfg)
        geom_proj = buffered_geom_in_crs(geom_4326, target_crs, aoi.get("buffer_m", 0))
        bbox_ll = latlon_bounds(geom_proj, target_crs)
        transform, width, height = build_target_grid(
            geom_proj, grid_cfg["resolution_m"], grid_cfg.get("snap_origin", True))
        log.info("=== AOI: %s (%s) | grid=%dx%d @ %sm ===", aoi_id, aoi.get("name", ""),
                 width, height, grid_cfg["resolution_m"])

        out_path = out_root / aoi_id / f"{aoi_id}.nc"
        if not overwrite and out_path.exists():
            log.info("  already processed, skipping")
            continue
        if dry_run:
            log.info("  [dry-run] would fetch GMRT %s for bbox %s", layer,
                     tuple(round(b, 3) for b in bbox_ll))
            continue

        try:
            tif = fetch_gmrt(bbox_ll, pad, layer, resolution, tmp_dir / f"{aoi_id}.tif")
            da = rioxarray.open_rasterio(tif, masked=True)
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)
            elev = da.rio.reproject(dst_crs=target_crs, shape=(height, width),
                                    transform=transform, resampling=Resampling.bilinear,
                                    nodata=np.nan)
            elev = elev.rio.clip([geom_proj], target_crs, drop=False)
            tif.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("  skipping %s (%s)", aoi_id, exc)
            continue

        depth = xr.where(elev < 0, -elev, 0.0).astype("float32")
        ds = xr.Dataset({"elevation": elev.astype("float32"), "depth": depth})
        ds["elevation"].attrs.update(units="m", long_name="elevation (neg below sea level)")
        ds["depth"].attrs.update(units="m", long_name="water depth (0 on land)")
        ds.attrs.update(aoi_id=aoi_id, source=f"GMRT GridServer ({layer}, {resolution})",
                        processing="reprojected/clipped to AOI grid")
        log.info("  wrote %s", write_output(ds, out_root / aoi_id, aoi_id, fmt))
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR bathymetry (GMRT) acquisition.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", help="Process only this AOI id.")
    ap.add_argument("--dry-run", action="store_true")
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