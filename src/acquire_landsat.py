#!/usr/bin/env python3
"""
OCEANSR -- Landsat thermal (surface temperature) acquisition via Google Earth
Engine.

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; Landsat settings come from `sources.landsat`. For each AOI and
platform it queries Landsat Collection 2, Tier 1, Level 2, builds a 3-band image
(sst in K + Landsat's OWN cloud and water masks), downloads just the AOI window
via getDownloadURL(), reprojects onto the AOI grid (identical to the ECOSTRESS
grid), and writes one aligned NetCDF per scene into data/LANDSAT/aligned/.

Landsat carries its own cloud (QA_PIXEL) and water (NDWI) masks because it rarely
has a coincident ECOSTRESS overpass -- so the two streams share a schema but each
brings independent masks. A later stage bins these to the daily datacube.

Usage (run from the OCEANSR project root, after `earthengine authenticate`):
    python src/acquire_landsat.py --config configs/config.yaml
    python src/acquire_landsat.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_landsat.py --config configs/config.yaml --dry-run
    python src/acquire_landsat.py --config configs/config.yaml --platforms LANDSAT_8
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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

log = logging.getLogger("acquire_landsat")
SOURCE = "landsat"

# Collection 2, Tier 1, Level 2 collections, and the ST / SR bands per platform.
COLLECTIONS = {
    "LANDSAT_5": "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT_7": "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT_8": "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT_9": "LANDSAT/LC09/C02/T1_L2",
}
BANDS = {
    "LANDSAT_5": {"st": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
    "LANDSAT_7": {"st": "ST_B6",  "green": "SR_B2", "nir": "SR_B4"},
    "LANDSAT_8": {"st": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
    "LANDSAT_9": {"st": "ST_B10", "green": "SR_B3", "nir": "SR_B5"},
}
# Collection 2 Level 2 scale/offset (USGS): ST -> Kelvin; SR -> reflectance.
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
CDIST_SCALE = 0.01  # ST_CDIST DN -> km


# --------------------------------------------------------------------------- #
# Config plumbing (mirrors acquire_ecostress.py)
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
    proj = cfg.get("project", {})
    return {
        "gee_project": proj.get("gee_project"),
        "gee_service_account": proj.get("gee_service_account"),
        "gee_key_file": proj.get("gee_key_file"),
        "aois": cfg["aois"],
        "time": time_cfg,
        "grid": grid_cfg,
        "ds": src_cfg,
        "out_dir": src_dir / "aligned",
        "fmt": src_cfg.get("output_format", "netcdf"),
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry / grid helpers (same conventions as the ECOSTRESS stage)
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
    """Lon/lat bounds of the (buffered) AOI, for the GEE query/region."""
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
        log.info("  using service account %s", service_account)
        creds = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)


def query_scenes(platform: str, rect_ll, start: str, end: str, cloud_max: float):
    """Return [(asset_id, time_start_ms), ...] for scenes over the AOI."""
    region = ee.Geometry.Rectangle(list(rect_ll))
    coll = (ee.ImageCollection(COLLECTIONS[platform])
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_max * 100.0)))
    info = coll.reduceColumns(
        ee.Reducer.toList().repeat(2), ["system:index", "system:time_start"]
    ).getInfo()["list"]
    idxs, times = (info[0], info[1]) if info else ([], [])
    cid = COLLECTIONS[platform]
    return [(f"{cid}/{idx}", int(t)) for idx, t in zip(idxs, times)]


def build_layers(asset_id: str, platform: str, mask_cfg: dict, to_celsius: bool) -> ee.Image:
    """3-band ee.Image: sst (K or degC), cloud (1=cloud), water (1=water)."""
    img = ee.Image(asset_id)
    b = BANDS[platform]

    kelvin = img.select(b["st"]).multiply(ST_SCALE).add(ST_OFFSET)
    sst = (kelvin.subtract(273.15) if to_celsius else kelvin).rename("sst").toFloat()

    # Cloud: QA_PIXEL dilated(1) | cloud(3) | shadow(4), plus ST_CDIST buffer.
    qa = img.select("QA_PIXEL")
    cloudy = (qa.bitwiseAnd(1 << 1).neq(0)
              .Or(qa.bitwiseAnd(1 << 3).neq(0))
              .Or(qa.bitwiseAnd(1 << 4).neq(0)))
    buf_km = float(mask_cfg.get("cloud_buffer_km", 1.0))
    if buf_km > 0:
        cdist_km = img.select("ST_CDIST").multiply(CDIST_SCALE)
        cloudy = cloudy.Or(cdist_km.lt(buf_km))
    cloud = cloudy.rename("cloud").toFloat()

    # Water from NDWI = (green - nir) / (green + nir).
    green = img.select(b["green"]).multiply(SR_SCALE).add(SR_OFFSET)
    nir = img.select(b["nir"]).multiply(SR_SCALE).add(SR_OFFSET)
    ndwi = green.subtract(nir).divide(green.add(nir))
    water = ndwi.gte(float(mask_cfg.get("ndwi_threshold", 0.0))).rename("water").toFloat()

    return sst.addBands(cloud).addBands(water)


def download_geotiff(image: ee.Image, rect_ll, target_crs: str, scale_m: float,
                     tmp_path: Path) -> Path:
    """getDownloadURL a multiband GeoTIFF for the AOI window; save to tmp_path."""
    region = ee.Geometry.Rectangle(list(rect_ll))
    url = image.getDownloadURL({
        "name": tmp_path.stem,
        "scale": scale_m,
        "crs": target_crs,
        "region": region,
        "format": "GEO_TIFF",
        "bands": ["sst", "cloud", "water"],
    })
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            content = r.content
            if content[:4] == b"PK\x03\x04":  # zipped -> extract first .tif
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    name = next(n for n in zf.namelist() if n.endswith(".tif"))
                    content = zf.read(name)
            tmp_path.write_bytes(content)
            return tmp_path
        except Exception as exc:
            last = exc
            time.sleep(4)
    raise RuntimeError(f"download failed after 3 attempts: {last}")


# --------------------------------------------------------------------------- #
# Align downloaded GeoTIFF to the AOI grid
# --------------------------------------------------------------------------- #
def align_to_grid(tif_path: Path, grid_cfg, target_crs, transform, width, height,
                  geom_proj, acq_time, aoi_id, platform) -> Optional[xr.Dataset]:
    da = rioxarray.open_rasterio(tif_path, masked=True)  # (band, y, x), bands sst/cloud/water
    names = ["sst", "cloud", "water"]
    rs_cont = Resampling[grid_cfg.get("resampling_continuous", "bilinear")]
    rs_cat = Resampling[grid_cfg.get("resampling_categorical", "nearest")]

    data_vars = {}
    for i, name in enumerate(names):
        layer = da.isel(band=i, drop=True)
        rs = rs_cat if name in ("cloud", "water") else rs_cont
        layer = layer.rio.reproject(dst_crs=target_crs, shape=(height, width),
                                    transform=transform, resampling=rs, nodata=np.nan)
        data_vars[name] = layer.rio.clip([geom_proj], target_crs, drop=False)

    ds = xr.Dataset(data_vars)
    ds["sst"].attrs["units"] = "degC" if grid_cfg.get("to_celsius", False) else "K"
    # Masks may carry NaN outside the AOI; treat NaN as 0 for the logic mask.
    valid = (np.isfinite(ds["sst"])
             & (ds["water"].fillna(0) > 0)
             & ~(ds["cloud"].fillna(0) > 0))
    ds["valid"] = valid.astype("uint8")
    ds["valid"].attrs["long_name"] = "water & clear & finite SST"

    ds = ds.expand_dims(time=[pd.Timestamp(acq_time)])
    ds.attrs.update(aoi_id=aoi_id, source=f"Landsat C2 L2 ST ({platform})",
                    processing="GEE getDownloadURL -> reprojected/clipped to AOI grid")
    return ds


def write_output(ds: xr.Dataset, out_dir: Path, aoi_id: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.Timestamp(ds["time"].values[0]).strftime("%Y%m%dT%H%M%S")
    stem = f"{aoi_id}_{t}"
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
def run(eff: dict, only_aoi, platforms_override, dry_run, max_scenes, tmp_dir: Path):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    mask_cfg = ds_cfg.get("masking", {})
    platforms = platforms_override or ds_cfg["platforms"]
    cloud_max = ds_cfg.get("cloud_cover_max", 0.7)
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]
    to_celsius = grid_cfg.get("to_celsius", False)

    init_gee(eff["gee_project"], eff.get("gee_service_account"), eff.get("gee_key_file"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    aois = eff["aois"]
    if only_aoi:
        aois = [a for a in aois if a["id"] == only_aoi]
        if not aois:
            raise SystemExit(f"AOI '{only_aoi}' not found in config.")

    n_done = 0
    for aoi in aois:
        aoi_id = aoi["id"]
        log.info("=== AOI: %s (%s) ===", aoi_id, aoi.get("name", ""))
        geom_4326 = aoi_geometry_4326(aoi)
        target_crs = resolve_target_crs(geom_4326, grid_cfg)
        geom_proj = buffered_geom_in_crs(geom_4326, target_crs, aoi.get("buffer_m", 0))
        rect_ll = latlon_rect(geom_proj, target_crs)
        transform, width, height = build_target_grid(
            geom_proj, grid_cfg["resolution_m"], grid_cfg.get("snap_origin", True))
        log.info("  target CRS=%s grid=%dx%d @ %sm", target_crs, width, height,
                 grid_cfg["resolution_m"])
        aoi_out = out_root / aoi_id

        scenes = []
        for plat in platforms:
            s = query_scenes(plat, rect_ll, start, end, cloud_max)
            log.info("  %s: %d scene(s) (cloud < %.0f%%)", plat, len(s), cloud_max * 100)
            scenes += [(plat, aid, t) for aid, t in s]
        scenes.sort(key=lambda x: x[2])

        if dry_run:
            log.info("  [dry-run] %d Landsat scene(s) for %s", len(scenes), aoi_id)
            continue

        for plat, asset_id, t_ms in scenes:
            acq = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
            tstr = acq.strftime("%Y%m%dT%H%M%S")
            if not overwrite and (aoi_out / f"{aoi_id}_{tstr}.nc").exists():
                log.info("  %s already processed, skipping", tstr)
                continue
            try:
                img = build_layers(asset_id, plat, mask_cfg, to_celsius)
                tif = download_geotiff(img, rect_ll, target_crs, grid_cfg["resolution_m"],
                                       tmp_dir / f"{aoi_id}_{tstr}.tif")
                ds = align_to_grid(tif, grid_cfg, target_crs, transform, width, height,
                                   geom_proj, acq, aoi_id, plat)
                tif.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("    skipping %s %s (%s)", plat, tstr, exc)
                continue
            log.info("      wrote %s", write_output(ds, aoi_out, aoi_id, fmt))
            n_done += 1
            if max_scenes and n_done >= max_scenes:
                log.info("Reached --max-scenes=%d; stopping.", max_scenes)
                return
    log.info("Done. Wrote %d scene(s).", n_done)


def main():
    ap = argparse.ArgumentParser(description="OCEANSR Landsat thermal acquisition (GEE).")
    ap.add_argument("--config", required=True, help="Path to configs/config.yaml.")
    ap.add_argument("--aoi", help="Process only this AOI id.")
    ap.add_argument("--platforms", nargs="+", help="Override platforms, e.g. LANDSAT_8 LANDSAT_9.")
    ap.add_argument("--dry-run", action="store_true", help="Query only; no download.")
    ap.add_argument("--max-scenes", type=int, help="Stop after N scenes (test batches).")
    ap.add_argument("--tmp", default="data/LANDSAT/_tmp", help="Scratch dir for downloads.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    eff = build_effective(cfg, SOURCE)
    run(eff, args.aoi, args.platforms, args.dry_run, args.max_scenes, Path(args.tmp))


if __name__ == "__main__":
    main()