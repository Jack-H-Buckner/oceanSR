#!/usr/bin/env python3
"""
OCEANSR -- bathymetry static covariate (NOAA NCEI CUDEM, GMRT fallback).

Settings come from `sources.bathymetry`. `source: cudem` reads the NOAA NCEI
Continuously Updated DEM (1/9 arc-second ~3 m seamless topobathy) straight from
its /vsicurl VRT, aggregates the fine pixels within each 100 m grid cell to
depth statistics, and (where CUDEM has no coverage, e.g. SE Alaska) falls back
to `source: gmrt` (GMRT GridServer, ~100 m). Writes ONE static NetCDF per AOI:

    data/BATHYMETRY/aligned/<aoi_id>/<aoi_id>.nc

Variables (all m):
  elevation  : mean elevation (neg below sea level) -- used by the landmask
  depth      : mean water depth over the cell (= mean of max(-elev,0))
  depth_p25  : 25th-percentile depth within the cell (sub-grid variability)
  depth_p75  : 75th-percentile depth within the cell

For GMRT (no sub-grid) depth_p25 = depth_p75 = depth. CUDEM is referenced to
NAVD88, not MSL -- expect the 0 contour (and water_min_depth_m) to shift slightly.

Usage (from the project root):
    python src/acquire_bathymetry.py --config configs/config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import warnings
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


def fine_grid(transform, width, height, k):
    """Sub-grid aligned to the coarse grid: same origin, resolution / k."""
    r = transform.a / k
    return from_origin(transform.c, transform.f, r, r), width * k, height * k


def block_stats(elev_fine, k, H, W):
    """(H*k, W*k) fine elevation -> per-coarse-cell (elev_mean, depth stats)."""
    ef = elev_fine.reshape(H, k, W, k)
    depth_fine = np.where(np.isnan(elev_fine), np.nan,
                          np.where(elev_fine < 0, -elev_fine, 0.0)).reshape(H, k, W, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN cells -> NaN
        elev_mean = np.nanmean(ef, axis=(1, 3))
        d_mean = np.nanmean(depth_fine, axis=(1, 3))
        d_p25 = np.nanpercentile(depth_fine, 25, axis=(1, 3))
        d_p75 = np.nanpercentile(depth_fine, 75, axis=(1, 3))
    return tuple(a.astype("float32") for a in (elev_mean, d_mean, d_p25, d_p75))


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def fetch_gmrt(bbox_ll, pad, layer, resolution, tmp_path: Path) -> Path:
    w, s, e, n = bbox_ll
    params = {"west": w - pad, "east": e + pad, "south": s - pad, "north": n + pad,
              "format": "geotiff", "layer": layer, "resolution": resolution}
    r = requests.get(GMRT_URL, params=params, timeout=180)
    r.raise_for_status()
    if r.content[:2] not in (b"II", b"MM"):
        raise RuntimeError(f"GMRT did not return a GeoTIFF (got {r.content[:80]!r})")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(r.content)
    return tmp_path


# CUDEM 1/9" tiles are 0.25-deg COGs named by their NW corner, e.g.
# ncei19_n47x75_w122x50_...tif -> lat 47.50-47.75, lon -122.50..-122.25.
CUDEM_URLLIST = ("https://coast.noaa.gov/htdata/raster2/elevation/"
                 "NCEI_ninth_Topobathy_2014_8483/urllist8483.txt")
_TILE_RE = re.compile(r"ncei19_n(\d+)x(\d+)_w(\d+)x(\d+)_", re.IGNORECASE)
CUDEM_NATIVE_M = 3.0


def _fetch_index(urllist, cache: Path):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(requests.get(urllist, timeout=60).text)
    return [u.strip() for u in cache.read_text().splitlines() if u.strip().endswith(".tif")]


def _tile_bounds(name):
    m = _TILE_RE.search(name)
    if not m:
        return None
    top = int(m.group(1)) + int(m.group(2)) / 100.0
    left = -(int(m.group(3)) + int(m.group(4)) / 100.0)
    return (left, top - 0.25, left + 0.25, top)          # (W, S, E, N)


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _choose_overview(target_m):
    """Map a target resolution to a COG overview level (None = full ~3 m)."""
    if target_m <= CUDEM_NATIVE_M * 1.5:
        return None
    return max(0, min(3, round(math.log2(target_m / CUDEM_NATIVE_M)) - 1))


def read_cudem(bbox_ll, target_crs, ftransform, Wf, Hf, geom_proj, urllist, cache, target_res_m):
    """Window-read ONLY the CUDEM tiles overlapping the AOI (COG overviews via
    /vsicurl + clip_box, so no full-mosaic allocation), merge, and reproject onto
    the fine sub-grid. Returns a (Hf, Wf) elevation array (NaN off-cover)."""
    from rioxarray.merge import merge_arrays
    urls = _fetch_index(urllist, Path(cache))
    sel = [u for u in urls if (tb := _tile_bounds(u)) and _overlaps(tb, bbox_ll)]
    if not sel:
        raise RuntimeError(f"no CUDEM tiles overlap bbox "
                           f"{tuple(round(b, 3) for b in bbox_ll)} ({len(urls)} in index)")
    ovr = _choose_overview(target_res_m)
    arrays = []
    for u in sel:
        da = None
        for lvl in dict.fromkeys([ovr, None]):           # requested overview, else full-res
            try:
                da = rioxarray.open_rasterio("/vsicurl/" + u, masked=True, overview_level=lvl)
                break
            except Exception:
                da = None
        if da is None:
            continue
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        tb = _tile_bounds(u)
        clip = (max(bbox_ll[0], tb[0]), max(bbox_ll[1], tb[1]),
                min(bbox_ll[2], tb[2]), min(bbox_ll[3], tb[3]))
        try:
            arrays.append(da.rio.clip_box(*clip))        # windowed read of the AOI portion
        except Exception:
            continue
    if not arrays:
        raise RuntimeError("all overlapping CUDEM tiles failed to read")
    mosaic = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
    fine = mosaic.rio.reproject(dst_crs=target_crs, shape=(Hf, Wf), transform=ftransform,
                                resampling=Resampling.nearest, nodata=np.nan)
    fine = fine.rio.clip([geom_proj], target_crs, drop=False)
    return fine.values.astype("float32")


def from_gmrt(bbox_ll, pad, layer, resolution, target_crs, transform, W, H, geom_proj, tmp):
    tif = fetch_gmrt(bbox_ll, pad, layer, resolution, tmp)
    da = rioxarray.open_rasterio(tif, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    elev = da.rio.reproject(dst_crs=target_crs, shape=(H, W), transform=transform,
                            resampling=Resampling.bilinear, nodata=np.nan)
    elev = elev.rio.clip([geom_proj], target_crs, drop=False).values.astype("float32")
    tif.unlink(missing_ok=True)
    depth = np.where(np.isnan(elev), np.nan,
                     np.where(elev < 0, -elev, 0.0)).astype("float32")
    return elev, depth, depth.copy(), depth.copy()          # no sub-grid -> p25=p75=mean


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
    source = str(ds_cfg.get("source", "gmrt")).lower()
    pad = float(ds_cfg.get("pad_deg", 0.02))
    layer = ds_cfg.get("layer", "topo")
    resolution = ds_cfg.get("resolution", "max")
    cudem_urllist = ds_cfg.get("cudem_urllist", CUDEM_URLLIST)
    cudem_cache = Path(ds_cfg.get("cudem_index_cache")) if ds_cfg.get("cudem_index_cache") \
        else out_root.parent / "urllist_cudem.txt"
    sub_m = float(ds_cfg.get("stats_subgrid_m", 10.0))
    min_cover = float(ds_cfg.get("min_cudem_cover", 0.5))
    res_m = float(grid_cfg["resolution_m"])
    k = max(1, int(round(res_m / sub_m)))

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
            geom_proj, res_m, grid_cfg.get("snap_origin", True))
        xs = transform.c + (np.arange(width) + 0.5) * transform.a
        ys = transform.f - (np.arange(height) + 0.5) * transform.a
        log.info("=== AOI: %s (%s) | grid=%dx%d @ %sm | source=%s ===",
                 aoi_id, aoi.get("name", ""), width, height, res_m, source)

        out_path = out_root / aoi_id / f"{aoi_id}.nc"
        if not overwrite and out_path.exists():
            log.info("  already processed, skipping"); continue
        if dry_run:
            log.info("  [dry-run] would build bathymetry (%s) for %s", source, aoi_id); continue

        elev = depth = dp25 = dp75 = None
        used = None
        if source == "cudem":
            try:
                ftr, Wf, Hf = fine_grid(transform, width, height, k)
                elev_fine = read_cudem(bbox_ll, target_crs, ftr, Wf, Hf, geom_proj,
                                       cudem_urllist, cudem_cache, sub_m)
                cover = float(np.isfinite(elev_fine).mean())
                if cover >= min_cover:
                    elev, depth, dp25, dp75 = block_stats(elev_fine, k, height, width)
                    used = f"NCEI CUDEM 1/9\" ({cover:.0%} cover, {k}x{k} subgrid)"
                else:
                    log.info("  %s: CUDEM cover %.0f%% < %.0f%% -> GMRT fallback",
                             aoi_id, 100 * cover, 100 * min_cover)
            except Exception as exc:
                log.warning("  %s: CUDEM read failed (%s) -> GMRT fallback", aoi_id, exc)

        if elev is None:                                     # GMRT (default or fallback)
            try:
                elev, depth, dp25, dp75 = from_gmrt(
                    bbox_ll, pad, layer, resolution, target_crs, transform,
                    width, height, geom_proj, tmp_dir / f"{aoi_id}.tif")
                used = f"GMRT ({layer}, {resolution})"
            except Exception as exc:
                log.warning("  skipping %s (%s)", aoi_id, exc); continue

        ds = xr.Dataset(
            {"elevation": (("y", "x"), elev), "depth": (("y", "x"), depth),
             "depth_p25": (("y", "x"), dp25), "depth_p75": (("y", "x"), dp75)},
            coords={"y": ys, "x": xs})
        ds["elevation"].attrs.update(units="m", long_name="mean elevation (neg below sea level)")
        ds["depth"].attrs.update(units="m", long_name="mean water depth (0 on land)")
        ds["depth_p25"].attrs.update(units="m", long_name="25th-percentile depth in cell")
        ds["depth_p75"].attrs.update(units="m", long_name="75th-percentile depth in cell")
        ds = ds.rio.write_crs(target_crs)
        ds.attrs.update(aoi_id=aoi_id, source=used,
                        processing="aggregated to AOI grid (mean, p25, p75 depth per cell)")
        log.info("  wrote %s  [%s]", write_output(ds, out_root / aoi_id, aoi_id, fmt), used)
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR bathymetry (CUDEM/GMRT) acquisition.")
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
