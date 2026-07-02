#!/usr/bin/env python3
"""
OCEANSR -- static land/water classification via Google Earth Engine (ESA WorldCover).

A single global land-cover mosaic regridded to each AOI's 100 m grid, written as a
STATIC layer data/LANDCOVER/aligned/<aoi>/<aoi>.nc with two variables:

  * landcover : the WorldCover class code (10 tree, 20 shrub, 30 grass, 40 crop,
                50 built, 60 bare, 70 snow, 80 water, 90 wetland, 95 mangrove, 100 moss)
  * water     : 1 where the class is in `water_classes` (default [80]), else 0

The assembler uses `water` as an authoritative LAND override: pixels the classifier
calls non-water are forced to land even when bathymetry is below sea level -- which
fixes diked/reclaimed farmland (negative elevation but not actually water).

Usage (from the project root, after `earthengine authenticate`):
    python src/acquire_landcover.py --config configs/config.yaml
    python src/acquire_landcover.py --config configs/config.yaml --aoi padilla_bay
    python src/acquire_landcover.py --config configs/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import xarray as xr
import yaml

import ee
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

log = logging.getLogger("acquire_landcover")
SOURCE = "landcover"


# --------------------------------------------------------------------------- #
# Config plumbing (mirrors acquire_landsat.py)
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
    src_dir = root / paths["data"] / paths.get(source, source.upper())
    proj = cfg.get("project", {})
    return {
        "gee_project": proj.get("gee_project"),
        "gee_service_account": proj.get("gee_service_account"),
        "gee_key_file": proj.get("gee_key_file"),
        "aois": cfg["aois"],
        "grid": grid_cfg,
        "ds": src_cfg,
        "out_dir": src_dir / "aligned",
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry / grid helpers (same conventions as the ECOSTRESS/Landsat stages)
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


def latlon_rect(geom_proj, target_crs: str):
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
# Earth Engine
# --------------------------------------------------------------------------- #
def init_gee(project: Optional[str], service_account: Optional[str] = None,
             key_file: Optional[str] = None):
    log.info("Initialising Google Earth Engine (project=%s)", project)
    if service_account and key_file:
        creds = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)


def build_image(asset: str, band: str, water_classes) -> ee.Image:
    """Static 2-band image: landcover class + binary water (1 in water_classes)."""
    mp = ee.ImageCollection(asset).mosaic().select(band)        # global mosaic of tiles
    water = mp.remap(list(water_classes), [1] * len(water_classes), 0).rename("water").toFloat()
    return mp.rename("landcover").toFloat().addBands(water)


def download_geotiff(image: ee.Image, rect_ll, target_crs: str, scale_m: float,
                     tmp_path: Path) -> Path:
    region = ee.Geometry.Rectangle(list(rect_ll))
    url = image.getDownloadURL({
        "name": tmp_path.stem, "scale": scale_m, "crs": target_crs,
        "region": region, "format": "GEO_TIFF", "bands": ["landcover", "water"],
    })
    last = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            content = r.content
            if content[:4] == b"PK\x03\x04":                    # zipped -> first .tif
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    name = next(n for n in zf.namelist() if n.endswith(".tif"))
                    content = zf.read(name)
            tmp_path.write_bytes(content)
            return tmp_path
        except Exception as exc:
            last = exc
            time.sleep(4)
    raise RuntimeError(f"download failed after 3 attempts: {last}")


def align_to_grid(tif_path, target_crs, transform, width, height, geom_proj, aoi_id) -> xr.Dataset:
    da = rioxarray.open_rasterio(tif_path, masked=True)         # (band, y, x): landcover, water
    out = {}
    for i, name in enumerate(("landcover", "water")):
        layer = da.isel(band=i, drop=True).rio.reproject(       # categorical -> nearest
            dst_crs=target_crs, shape=(height, width), transform=transform,
            resampling=Resampling.nearest, nodata=np.nan)
        out[name] = layer.rio.clip([geom_proj], target_crs, drop=False)
    ds = xr.Dataset(out)
    ds["water"] = ds["water"].fillna(0).astype("uint8")          # 1=water, 0=land/unknown
    ds["water"].attrs["long_name"] = "ESA WorldCover water class (1=water)"
    ds.attrs.update(aoi_id=aoi_id, source="ESA WorldCover (GEE mosaic)")
    return ds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff, only_aoi, dry_run, tmp_dir: Path):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, overwrite = eff["out_dir"], eff["overwrite"]
    asset = ds_cfg.get("asset", "ESA/WorldCover/v200")
    band = ds_cfg.get("band", "Map")
    water_classes = ds_cfg.get("water_classes", [80])
    res = grid_cfg["resolution_m"]

    init_gee(eff["gee_project"], eff.get("gee_service_account"), eff.get("gee_key_file"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image = build_image(asset, band, water_classes)

    aois = eff["aois"]
    if only_aoi:
        aois = [a for a in aois if a["id"] == only_aoi]
        if not aois:
            raise SystemExit(f"AOI '{only_aoi}' not found in config.")

    for aoi in aois:
        aid = aoi["id"]
        out_f = out_root / aid / f"{aid}.nc"
        if out_f.exists() and not overwrite:
            log.info("=== %s: %s exists, skipping (use overwrite) ===", aid, out_f.name)
            continue
        geom_4326 = aoi_geometry_4326(aoi)
        target_crs = resolve_target_crs(geom_4326, grid_cfg)
        geom_proj = buffered_geom_in_crs(geom_4326, target_crs, aoi.get("buffer_m", 0))
        rect_ll = latlon_rect(geom_proj, target_crs)
        transform, width, height = build_target_grid(
            geom_proj, res, grid_cfg.get("snap_origin", True))
        log.info("=== AOI: %s | %s | grid=%dx%d @ %sm | water_classes=%s ===",
                 aid, asset, width, height, res, water_classes)
        if dry_run:
            continue
        tif = download_geotiff(image, rect_ll, target_crs, res, tmp_dir / f"{aid}_lc.tif")
        ds = align_to_grid(tif, target_crs, transform, width, height, geom_proj, aid)
        tif.unlink(missing_ok=True)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_f, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
        wf = float((ds["water"].values == 1).mean())
        log.info("  wrote %s  (water=%.0f%% of grid)", out_f, 100 * wf)
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR static land-cover acquisition (GEE).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", help="Process only this AOI id.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tmp", default="data/LANDCOVER/_tmp")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    eff = build_effective(cfg, SOURCE)
    run(eff, args.aoi, args.dry_run, Path(args.tmp))


if __name__ == "__main__":
    main()