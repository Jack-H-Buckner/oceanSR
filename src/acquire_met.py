#!/usr/bin/env python3
"""
OCEANSR -- meteorological forcing acquisition (NOAA HRRR via Herbie).

Reads the common project config (configs/config.yaml): AOIs, grid, dates and
paths are shared; HRRR settings come from `sources.met`. For each AOI it pulls
2 m air temperature, 10 m wind (u/v + speed) and downward shortwave, regrids the
HRRR field onto the AOI grid (identical to the SST stages), and writes:

  * a daily-mean file per day:           data/MET/aligned/<aoi>/<aoi>_<YYYYMMDD>.nc
  * an instantaneous file per overpass:  data/MET/aligned/<aoi>/<aoi>_<YYYYMMDDThhmmss>.nc

Overpass times are discovered from the ECOSTRESS/Landsat aligned dirs so the
forcing can be matched to each thermal scene (for the skin->bulk correction).
The CONUS domain ('hrrr') is used below ~50N; SE Alaska AOIs use 'hrrrak'.

Forcing is gap-free, so there is no mask -- these are complete driver channels.

Usage (run AFTER the SST stages, from the project root):
    python src/acquire_met.py --config configs/config.yaml --aoi hood_canal
    python src/acquire_met.py --config configs/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from herbie import Herbie
from pyproj import Transformer
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.kd_tree import resample_nearest
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

log = logging.getLogger("acquire_met")
SOURCE = "met"
_DT_RE = re.compile(r"(\d{8}T\d{6})")

# config var key -> (Herbie search regex, {cfgrib var name: output name})
VAR_SEARCH = {
    "airtemp": (r"TMP:2 m above ground", {"t2m": "airtemp"}),
    "wind":    (r":(U|V)GRD:10 m above ground", {"u10": "wind_u", "v10": "wind_v"}),
    "swrad":   (r"DSWRF:surface", {"dswrf": "swrad"}),
}


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
    time_cfg = {**cfg.get("time", {}), **src_cfg.get("time", {})}
    grid_cfg = {**cfg.get("grid", {}), **src_cfg.get("grid", {})}
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    return {
        "aois": cfg["aois"],
        "time": time_cfg,
        "grid": grid_cfg,
        "ds": src_cfg,
        "root": root,
        "paths": paths,
        "out_dir": root / paths["data"] / paths[source] / "aligned",
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


def make_area(target_crs, transform, width, height) -> AreaDefinition:
    res = transform.a
    minx, maxy = transform.c, transform.f
    extent = (minx, maxy - height * res, minx + width * res, maxy)  # (xmin,ymin,xmax,ymax)
    return AreaDefinition("aoi", "aoi", "aoi", target_crs, width, height, extent)


# --------------------------------------------------------------------------- #
# HRRR fetch + regrid
# --------------------------------------------------------------------------- #
def model_for_aoi(geom_4326, configured: str) -> str:
    if configured and configured != "auto":
        return configured
    return "hrrrak" if geom_4326.centroid.y >= 50.0 else "hrrr"


def snap_to_cycle(dt: datetime, model: str) -> datetime:
    """Round to the nearest available analysis cycle (1h for hrrr, 3h for hrrrak)."""
    step = 3 if model == "hrrrak" else 1
    h = int(round(dt.hour / step) * step)
    base = dt.replace(minute=0, second=0, microsecond=0, hour=0)
    return base + timedelta(hours=min(h, 24 - step))


def fetch_hrrr(model, dt, fxx, product, var_keys):
    """Return ({outname: 2D array}, lon2d, lat2d) for one cycle, or None on miss."""
    H = Herbie(dt.strftime("%Y-%m-%d %H:00"), model=model, product=product, fxx=fxx)
    fields, lon2d, lat2d = {}, None, None
    for key in var_keys:
        search, rename = VAR_SEARCH[key]
        ds = H.xarray(search, remove_grib=True)
        if isinstance(ds, list):
            ds = xr.merge(ds, compat="override")
        for src, dst in rename.items():
            if src in ds:
                fields[dst] = np.asarray(ds[src].values)
        if lon2d is None and "longitude" in ds.coords:
            lon2d = np.asarray(ds.longitude.values)
            lat2d = np.asarray(ds.latitude.values)
    if not fields or lon2d is None:
        return None
    lon2d = ((lon2d + 180.0) % 360.0) - 180.0  # 0..360 -> -180..180
    return fields, lon2d, lat2d


def regrid(fields, lon2d, lat2d, area, radius_m) -> dict:
    swath = SwathDefinition(lons=lon2d, lats=lat2d)
    out = {}
    for name, arr in fields.items():
        out[name] = resample_nearest(swath, arr, area, radius_of_influence=radius_m,
                                     fill_value=np.nan)
    return out


def to_dataset(grids, coords, t, grid_cfg) -> xr.Dataset:
    """grids: {name: 2D array} on the AOI grid -> Dataset with wind_speed + time."""
    y, x = coords
    dv = {k: (("y", "x"), v.astype("float32")) for k, v in grids.items()}
    ds = xr.Dataset(dv, coords={"y": y, "x": x})
    if "wind_u" in ds and "wind_v" in ds:
        ds["wind_speed"] = np.sqrt(ds["wind_u"] ** 2 + ds["wind_v"] ** 2)
    if "airtemp" in ds:
        if grid_cfg.get("to_celsius", False):
            ds["airtemp"] = ds["airtemp"] - 273.15
            ds["airtemp"].attrs["units"] = "degC"
        else:
            ds["airtemp"].attrs["units"] = "K"
    if "wind_speed" in ds:
        ds["wind_speed"].attrs["units"] = "m s-1"
    if "swrad" in ds:
        ds["swrad"].attrs["units"] = "W m-2"
    return ds.expand_dims(time=[pd.Timestamp(t)])


# --------------------------------------------------------------------------- #
# Overpass discovery
# --------------------------------------------------------------------------- #
def overpass_times_for_day(root, paths, sources, aoi_id, day) -> list:
    """Datetimes (UTC, naive) of ECOSTRESS/Landsat scenes for this AOI on `day`."""
    daystr = day.strftime("%Y%m%d")
    times = []
    for src in sources:
        d = root / paths["data"] / paths[src] / "aligned" / aoi_id
        if not d.exists():
            continue
        for f in d.glob(f"{aoi_id}_{daystr}T*.nc"):
            m = _DT_RE.search(f.name)
            if m:
                times.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
    return sorted(set(times))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def write_output(ds, out_dir, aoi_id, fmt, stem) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
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


def run(eff, only_aoi, dry_run):
    ds_cfg, grid_cfg = eff["ds"], eff["grid"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    var_keys = ds_cfg.get("variables", ["airtemp", "wind", "swrad"])
    fxx = int(ds_cfg.get("fxx", 0))
    product = ds_cfg.get("product", "sfc")
    mean_hours = ds_cfg.get("daily_mean_hours", [0, 6, 12, 18])
    radius_m = float(ds_cfg.get("regrid_radius_m", 6000))
    op_sources = ds_cfg.get("overpass_from", [])
    start = pd.Timestamp(eff["time"]["start_date"])
    end = pd.Timestamp(eff["time"]["end_date"])
    days = pd.date_range(start, end, freq="D")

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
        transform, width, height = build_target_grid(
            geom_proj, grid_cfg["resolution_m"], grid_cfg.get("snap_origin", True))
        area = make_area(target_crs, transform, width, height)
        xs = transform.c + (np.arange(width) + 0.5) * transform.a
        ys = transform.f - (np.arange(height) + 0.5) * transform.a
        model = model_for_aoi(geom_4326, ds_cfg.get("model", "auto"))
        log.info("=== AOI: %s (%s) | model=%s grid=%dx%d ===",
                 aoi_id, aoi.get("name", ""), model, width, height)
        if dry_run:
            log.info("  [dry-run] %d days; daily hours=%s + overpass snapshots", len(days), mean_hours)
            continue
        aoi_out = out_root / aoi_id

        for day in days:
            dstr = day.strftime("%Y%m%d")
            # ---- daily mean over mean_hours ----
            if overwrite or not (aoi_out / f"{aoi_id}_{dstr}.nc").exists():
                stack = {}
                for hh in mean_hours:
                    dt = snap_to_cycle(day.to_pydatetime().replace(hour=int(hh)), model)
                    try:
                        got = fetch_hrrr(model, dt, fxx, product, var_keys)
                        if got is None:
                            continue
                        grids = regrid(got[0], got[1], got[2], area, radius_m)
                    except Exception as exc:
                        log.warning("    %s %02dZ fetch failed (%s)", dstr, hh, exc)
                        continue
                    for k, v in grids.items():
                        stack.setdefault(k, []).append(v)
                if stack:
                    mean_grids = {k: np.nanmean(np.stack(v), axis=0) for k, v in stack.items()}
                    ds = to_dataset(mean_grids, (ys, xs), day, grid_cfg)
                    ds.attrs.update(aoi_id=aoi_id, source=f"NOAA {model} daily mean",
                                    daily_mean_hours=str(mean_hours))
                    log.info("  %s daily -> %s", dstr,
                             write_output(ds, aoi_out, aoi_id, fmt, f"{aoi_id}_{dstr}"))

            # ---- overpass snapshots ----
            for op in overpass_times_for_day(eff["root"], eff["paths"], op_sources, aoi_id, day):
                tstr = op.strftime("%Y%m%dT%H%M%S")
                if not overwrite and (aoi_out / f"{aoi_id}_{tstr}.nc").exists():
                    continue
                dt = snap_to_cycle(op, model)
                try:
                    got = fetch_hrrr(model, dt, fxx, product, var_keys)
                    if got is None:
                        continue
                    grids = regrid(got[0], got[1], got[2], area, radius_m)
                except Exception as exc:
                    log.warning("    overpass %s fetch failed (%s)", tstr, exc)
                    continue
                ds = to_dataset(grids, (ys, xs), op, grid_cfg)
                ds.attrs.update(aoi_id=aoi_id, source=f"NOAA {model} @overpass",
                                cycle=dt.strftime("%Y-%m-%dT%H:00"))
                log.info("  overpass %s -> %s", tstr,
                         write_output(ds, aoi_out, aoi_id, fmt, f"{aoi_id}_{tstr}"))
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR HRRR meteorological forcing acquisition.")
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