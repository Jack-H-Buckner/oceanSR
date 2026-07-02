#!/usr/bin/env python3
"""
OCEANSR -- datacube assembler.

Knits the per-AOI aligned outputs (ECOSTRESS, Landsat, MUR, MET, bathymetry,
tide) into one analysis-ready, chunked Zarr cube per AOI on a common DAILY time
axis, and writes global normalization stats + a valid-tile index for the trainer.

Design choices (locked):
  * Zarr per AOI, chunked in (time, y, x).
  * SST kept SEPARATE per sensor (mur_sst / eco_sst / lst_sst) so the model's
    learned per-source offsets survive; each high-res sensor carries its own
    cloud + valid masks and overpass hour.
  * Multiple scenes of one sensor on a day -> keep the CLEAREST (most valid px).
  * Also emit norm_stats.json (per-channel mean/std over valid pixels) and
    tile_index.csv (tile origins with enough water).

Channel layout in each <aoi>.zarr:
  3D (time,y,x): mur_sst, mur_valid, eco_sst, eco_cloud, eco_valid,
                 lst_sst, lst_cloud, lst_valid, airtemp, wind_u, wind_v,
                 wind_speed, swrad
  2D (y,x) static: depth, landmask
  1D (time): tide, tide_range, eco_hour, lst_hour, doy_sin, doy_cos

Usage (run after all acquisition stages, from the project root):
    python src/assemble_datacube.py --config configs/config.yaml
    python src/assemble_datacube.py --config configs/config.yaml --aoi hood_canal
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

log = logging.getLogger("assemble_datacube")
_DT_RE = re.compile(r"(\d{8}T\d{6})")
_D_RE = re.compile(r"_(\d{8})\.nc$")

# Continuous channels that get z-score normalization stats.
CONTINUOUS = ["mur_sst", "eco_sst", "lst_sst", "airtemp", "wind_u", "wind_v",
              "wind_speed", "swrad", "tide", "tide_range", "depth",
              "depth_p25", "depth_p75"]


# --------------------------------------------------------------------------- #
# Config + grid
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def utm_epsg_from_lonlat(lon, lat):
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def aoi_geometry_4326(aoi):
    if aoi.get("geometry"):
        gj = json.load(open(aoi["geometry"]))
        if gj.get("type") == "FeatureCollection":
            return unary_union([shape(f["geometry"]) for f in gj["features"]])
        if gj.get("type") == "Feature":
            return shape(gj["geometry"])
        return shape(gj)
    w, s, e, n = aoi["bbox"]
    return box(w, s, e, n)


def canonical_grid(aoi, grid_cfg):
    """Return (target_crs, xs, ys) -- the same grid every acquisition stage used."""
    geom = aoi_geometry_4326(aoi)
    tc = str(grid_cfg.get("target_crs", "auto")).lower()
    crs = utm_epsg_from_lonlat(geom.centroid.x, geom.centroid.y) if tc == "auto" \
        else grid_cfg["target_crs"]
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    g = shp_transform(fwd, geom)
    if aoi.get("buffer_m", 0):
        g = g.buffer(aoi["buffer_m"])
    r = float(grid_cfg["resolution_m"])
    minx, miny, maxx, maxy = g.bounds
    if grid_cfg.get("snap_origin", True):
        minx, miny = math.floor(minx / r) * r, math.floor(miny / r) * r
        maxx, maxy = math.ceil(maxx / r) * r, math.ceil(maxy / r) * r
    W = int(round((maxx - minx) / r))
    H = int(round((maxy - miny) / r))
    xs = minx + (np.arange(W) + 0.5) * r
    ys = maxy - (np.arange(H) + 0.5) * r
    return crs, xs, ys


# --------------------------------------------------------------------------- #
# Loaders (each returns arrays on the daily axis / canonical grid)
# --------------------------------------------------------------------------- #
def _empty3d(days, H, W):
    return np.full((len(days), H, W), np.nan, dtype="float32")


def _aligned_dir(root, paths, source, aoi_id):
    # Default the sub-folder from the source name if not in paths (e.g. "tide"->"TIDE"),
    # so a missing paths entry (or a source not yet acquired) doesn't crash.
    sub = paths.get(source, source.upper())
    return root / paths["data"] / sub / "aligned" / aoi_id


def load_daily_sensor(d: Path, aoi_id, days, H, W, var):
    """MUR/MET style: one file per day (<aoi>_YYYYMMDD.nc). Returns {var: 3D}."""
    out = _empty3d(days, H, W)
    if not d.exists():
        return out
    didx = {dd.strftime("%Y%m%d"): i for i, dd in enumerate(days)}
    for f in d.glob(f"{aoi_id}_*.nc"):
        m = _D_RE.search(f.name)
        if not m or m.group(1) not in didx:
            continue
        ds = xr.open_dataset(f)
        if var in ds:
            arr = ds[var].isel(time=0).values if "time" in ds[var].dims else ds[var].values
            if arr.shape == (H, W):
                out[didx[m.group(1)]] = arr
        ds.close()
    return out


def load_clearest_overpass(d: Path, aoi_id, days, H, W, water_is_land=False,
                           qc_levels=None, use_cloud=True):
    """ECOSTRESS/Landsat: per-overpass files. Keep the clearest WATER scene per day.

    Validity = finite(sst) & water [& clear] [& QC-produced]:
      * water: sensor water layer with per-sensor polarity (`water_is_land`).
      * use_cloud=True gates on the binary cloud layer (Landsat: reliable).
      * qc_levels (e.g. {0,1}) gates on QC mandatory-QA bits 0-1 instead of cloud
        (ECOSTRESS: cloud over-masks cold water, so use QC).
    Returns `water_union` (OR of the water mask over scenes) -- a high-res static
    water mask that resolves narrow estuaries (unlike the coarse bathymetry DEM).
    """
    sst, cloud = _empty3d(days, H, W), _empty3d(days, H, W)
    valid = np.zeros((len(days), H, W), dtype="uint8")
    hour = np.full(len(days), np.nan, dtype="float32")
    water_union = np.zeros((H, W), dtype=bool)
    if not d.exists():
        return sst, cloud, valid, hour, water_union
    qset = list(qc_levels) if qc_levels is not None else None
    didx = {dd.strftime("%Y%m%d"): i for i, dd in enumerate(days)}
    best = {}  # day -> (count, sst, cloud, valid, datetime)
    for f in d.glob(f"{aoi_id}_*T*.nc"):
        m = _DT_RE.search(f.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        day = dt.strftime("%Y%m%d")
        if day not in didx:
            continue
        ds = xr.open_dataset(f)
        if "time" in ds.dims:
            ds = ds.isel(time=0)
        if "sst" not in ds or ds["sst"].shape != (H, W):
            ds.close(); continue
        s = ds["sst"].values.astype("float32")
        c = (ds["cloud"].values.astype("float32")
             if "cloud" in ds and ds["cloud"].shape == (H, W) else np.zeros((H, W), "float32"))
        if "water" in ds and ds["water"].shape == (H, W):
            w = ds["water"].values.astype("float32")
            wp = np.isfinite(w) & ((w < 0.5) if water_is_land else (w > 0.5))
        else:
            wp = np.zeros((H, W), dtype=bool)              # no water layer -> claim NOTHING
            #   (defaulting to "all water" would pollute the static water mask)
        q = (ds["quality"].values if "quality" in ds and ds["quality"].shape == (H, W) else None)
        ds.close()

        water_union |= wp
        v = np.isfinite(s) & wp
        if use_cloud:
            v &= ~(np.nan_to_num(c, nan=1.0) > 0)
        if qset is not None and q is not None:
            mqa = np.full((H, W), -1, dtype="int64")
            fin = np.isfinite(q)
            mqa[fin] = q[fin].astype("int64") & 0b11       # mandatory-QA bits 0-1
            v &= np.isin(mqa, qset)
        vc = int(v.sum())
        if day not in best or vc > best[day][0]:
            best[day] = (vc, s, c, v, dt)
    for day, (_, s, c, v, dt) in best.items():
        i = didx[day]
        sst[i] = s
        cloud[i] = np.nan_to_num(c, nan=0.0)
        valid[i] = v.astype("uint8")
        hour[i] = dt.hour + dt.minute / 60.0
    return sst, cloud, valid, hour, water_union


def load_tide_daily(d: Path, aoi_id, days):
    """Tide 1D series -> daily mean + daily range on the daily axis."""
    mean = np.full(len(days), np.nan, "float32")
    rng = np.full(len(days), np.nan, "float32")
    f = d / f"{aoi_id}_tides.nc"
    if not f.exists():
        return mean, rng
    ds = xr.open_dataset(f)
    t = ds["tide"]
    dm = t.resample(time="1D").mean()
    dr = t.resample(time="1D").max() - t.resample(time="1D").min()
    dm = dm.assign_coords(time=dm["time"].dt.strftime("%Y%m%d").values)
    dr = dr.assign_coords(time=dr["time"].dt.strftime("%Y%m%d").values)
    lut_m = dict(zip(dm["time"].values, dm.values))
    lut_r = dict(zip(dr["time"].values, dr.values))
    for i, dd in enumerate(days):
        k = dd.strftime("%Y%m%d")
        if k in lut_m:
            mean[i] = lut_m[k]
            rng[i] = lut_r[k]
    ds.close()
    return mean, rng


def fill_water_nn(arr, water):
    """Nearest-neighbour fill of NaN values over `water` pixels, per time slice.

    For each day, water pixels with no MUR value take the value of the nearest
    finite MUR pixel (typically just-offshore open water). Land/non-water NaNs are
    left as-is. `arr` is (T,H,W); `water` is (H,W) bool.
    """
    from scipy.ndimage import distance_transform_edt
    out = arr.copy()
    for t in range(out.shape[0]):
        m = out[t]
        finite = np.isfinite(m)
        need = (~finite) & water
        if need.any() and finite.any():
            idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
            nn = m[tuple(idx)]                 # nearest finite value at every pixel
            m[need] = nn[need]
            out[t] = m
    return out


def load_bathy(d: Path, aoi_id, H, W):
    elev = np.full((H, W), np.nan, "float32")
    depth = dp25 = dp75 = None
    f = d / f"{aoi_id}.nc"
    if f.exists():
        ds = xr.open_dataset(f)
        def g(name):
            return (ds[name].values.astype("float32")
                    if name in ds and ds[name].shape == (H, W) else None)
        if g("elevation") is not None:
            elev = g("elevation")
        depth, dp25, dp75 = g("depth"), g("depth_p25"), g("depth_p75")
        ds.close()
    if depth is None:                                    # old file: derive mean depth
        depth = np.where(elev < 0, -elev, 0.0).astype("float32")
    if dp25 is None:
        dp25 = depth.copy()                              # no sub-grid stats available
    if dp75 is None:
        dp75 = depth.copy()
    return elev, depth, dp25, dp75


def load_landcover(d: Path, aoi_id, H, W):
    """Static ESA WorldCover water mask. Returns float (1=water, 0=land, NaN=unknown)."""
    water = np.full((H, W), np.nan, "float32")
    f = d / f"{aoi_id}.nc"
    if f.exists():
        ds = xr.open_dataset(f)
        if "water" in ds and ds["water"].shape == (H, W):
            water = ds["water"].values.astype("float32")
        ds.close()
    return water


# --------------------------------------------------------------------------- #
# Assemble one AOI
# --------------------------------------------------------------------------- #
def assemble_aoi(aoi, cfg, days) -> xr.Dataset:
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    crs, xs, ys = canonical_grid(aoi, {**cfg["grid"]})
    H, W = len(ys), len(xs)
    aid = aoi["id"]

    def adir(src):
        return _aligned_dir(root, paths, src, aid)

    elev, depth, depth_p25, depth_p75 = load_bathy(adir("bathymetry"), aid, H, W)
    srcs = cfg.get("sources", {})
    eco_cfg, lst_cfg = srcs.get("ecostress", {}), srcs.get("landsat", {})
    eco_wil = eco_cfg.get("water_is_land", True)        # ECOSTRESS layer: 1=land
    lst_wil = lst_cfg.get("water_is_land", False)       # Landsat NDWI: 1=water
    eco_qc = eco_cfg.get("qc_levels", [0, 1])           # ECOSTRESS: gate on QC, not cloud

    mur_sst = load_daily_sensor(adir("mur"), aid, days, H, W, "sst")
    # ECOSTRESS: water + QC-produced (cloud over-masks cold water, so don't gate on it)
    eco_sst, eco_cloud, eco_valid, eco_hour, eco_wu = load_clearest_overpass(
        adir("ecostress"), aid, days, H, W, water_is_land=eco_wil,
        qc_levels=eco_qc, use_cloud=False)
    # Landsat: water + cloud (its QA_PIXEL-based cloud is reliable)
    lst_sst, lst_cloud, lst_valid, lst_hour, lst_wu = load_clearest_overpass(
        adir("landsat"), aid, days, H, W, water_is_land=lst_wil,
        qc_levels=None, use_cloud=True)
    airtemp = load_daily_sensor(adir("met"), aid, days, H, W, "airtemp")
    wind_u = load_daily_sensor(adir("met"), aid, days, H, W, "wind_u")
    wind_v = load_daily_sensor(adir("met"), aid, days, H, W, "wind_v")
    wind_speed = load_daily_sensor(adir("met"), aid, days, H, W, "wind_speed")
    swrad = load_daily_sensor(adir("met"), aid, days, H, W, "swrad")
    cloud_cover = load_daily_sensor(adir("met"), aid, days, H, W, "cloud_cover")
    tide, tide_range = load_tide_daily(adir("tide"), aid, days)

    # static water mask = sensor water (resolves narrow estuaries) UNION bathymetry
    # water, then TWO elevation overrides (both require known bathymetry):
    #   - high_land: elevation above land_elev_threshold_m is always land
    #   - intertidal: elevation shallower than -water_min_depth_m (i.e. above the
    #     min-depth contour) is dropped -- excludes tide flats / always-exposed
    #     shoals so we keep only typically-submerged (subtidal) water.
    asm_cfg = cfg.get("assembler", {})
    min_depth = float(asm_cfg.get("water_min_depth_m", 0.0))   # 0 = disabled (elev<0)
    land_thr = float(asm_cfg.get("land_elev_threshold_m", 2.0))
    bathy_water = np.isfinite(elev) & (elev < -min_depth)
    high_land = np.isfinite(elev) & (elev > land_thr)
    static_water = (eco_wu | lst_wu | bathy_water) & ~high_land
    if min_depth > 0:                                          # drop intertidal/shoals
        intertidal = np.isfinite(elev) & (elev > -min_depth)
        removed = int((static_water & intertidal).sum())
        static_water &= ~intertidal
        log.info("  %s: water_min_depth_m=%.1f removed %d intertidal/shallow px "
                 "(where bathymetry is known)", aid, min_depth, removed)
    # land-cover: always load so we can (a) override the landmask and (b) expose the
    # raw WorldCover water mask as a cube variable usable as a per-pixel loss filter.
    lc_raw = load_landcover(adir("landcover"), aid, H, W)     # 1=water, 0=land, NaN=unknown
    lc_known = np.isfinite(lc_raw)
    # filter channel: 1 = water OR unknown (no-op), 0 = KNOWN land. unknown->water so a
    # "landcover_water == 1" filter can't wipe supervision where the layer is missing.
    landcover_water = np.where(lc_known, lc_raw > 0.5, True).astype("uint8")
    if asm_cfg.get("use_landcover_override", True):          # force land where known non-water
        lc_land = lc_known & (lc_raw < 0.5)
        removed = int((static_water & lc_land).sum())
        static_water &= ~lc_land
        log.info("  %s: land-cover override removed %d non-water px "
                 "(known land-cover classified as land)", aid, removed)
    landmask = (~static_water).astype("uint8")           # 1 = land
    wf = float(static_water.mean())
    if wf > 0.98:
        log.warning("  %s: static water is %.0f%% of the tile -- check the source water "
                    "layer / land_elev_threshold_m", aid, 100 * wf)

    # MUR is 1 km: narrow-estuary water pixels can be empty after upsampling.
    # Nearest-neighbour fill the backbone over water so it's present everywhere.
    if cfg.get("assembler", {}).get("fill_mur_water", True):
        mur_sst = fill_water_nn(mur_sst, static_water)
    mur_valid = np.isfinite(mur_sst).astype("uint8")

    # MUR cold-deviation cloud filter: drop high-res pixels colder than the MUR
    # backbone by > threshold K (clouds bias TIR cold). NaN diffs -> False (kept).
    thr = float(cfg.get("assembler", {}).get("mur_cloud_threshold_k", 0))
    if thr > 0:
        eco_valid = (eco_valid.astype(bool) & ~((mur_sst - eco_sst) > thr)).astype("uint8")
        lst_valid = (lst_valid.astype(bool) & ~((mur_sst - lst_sst) > thr)).astype("uint8")
    doy = days.dayofyear.values.astype("float32")
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    T = ("time", "y", "x")
    ds = xr.Dataset(
        {
            "mur_sst": (T, mur_sst), "mur_valid": (T, mur_valid),
            "eco_sst": (T, eco_sst), "eco_cloud": (T, eco_cloud), "eco_valid": (T, eco_valid),
            "lst_sst": (T, lst_sst), "lst_cloud": (T, lst_cloud), "lst_valid": (T, lst_valid),
            "airtemp": (T, airtemp), "wind_u": (T, wind_u), "wind_v": (T, wind_v),
            "wind_speed": (T, wind_speed), "swrad": (T, swrad), "cloud_cover": (T, cloud_cover),
            "depth": (("y", "x"), depth),
            "depth_p25": (("y", "x"), depth_p25), "depth_p75": (("y", "x"), depth_p75),
            "landmask": (("y", "x"), landmask),
            "landcover_water": (("y", "x"), landcover_water),
            "tide": (("time",), tide), "tide_range": (("time",), tide_range),
            "eco_hour": (("time",), eco_hour), "lst_hour": (("time",), lst_hour),
            "doy_sin": (("time",), doy_sin), "doy_cos": (("time",), doy_cos),
        },
        coords={"time": days, "y": ys, "x": xs},
    )
    ds.attrs.update(aoi_id=aid, name=aoi.get("name", ""), crs=crs,
                    region=aoi.get("region", ""))
    return ds


# --------------------------------------------------------------------------- #
# Stats + tile index (accumulated across AOIs)
# --------------------------------------------------------------------------- #
class StatsAccumulator:
    # SST channels carry full LST (land + water); restrict their stats to water
    # pixels (via the matching valid mask) so the SST scale isn't inflated by land.
    SST_MASK = {"eco_sst": "eco_valid", "lst_sst": "lst_valid", "mur_sst": "mur_valid"}

    def __init__(self, channels):
        self.s = {c: [0.0, 0.0, 0] for c in channels}  # sum, sumsq, count

    def update(self, ds):
        for c in self.s:
            if c not in ds:
                continue
            v = ds[c].values.astype("float64")
            mk = self.SST_MASK.get(c)
            if mk and mk in ds:
                v = v[(ds[mk].values > 0) & np.isfinite(v)]   # water only
            else:
                v = v[np.isfinite(v)]
            if v.size:
                self.s[c][0] += v.sum()
                self.s[c][1] += (v ** 2).sum()
                self.s[c][2] += v.size

    def result(self):
        out = {}
        for c, (s, ss, n) in self.s.items():
            if n > 0:
                mean = s / n
                var = max(ss / n - mean ** 2, 0.0)
                out[c] = {"mean": mean, "std": math.sqrt(var) or 1.0, "count": n}
        return out


def zarr_encoding(ds, chunks):
    """Per-variable chunk encoding (avoids needing dask to write)."""
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple(min(chunks.get(d, ds.sizes[d]), ds.sizes[d]) for d in dims)
        if ch:
            enc[v] = {"chunks": ch}
    return enc


def tile_origins(ds, size, stride, min_water_frac):
    water = (ds["landmask"].values == 0)  # True over water
    H, W = water.shape
    rows = []
    for y0 in range(0, max(H - size + 1, 1), stride):
        for x0 in range(0, max(W - size + 1, 1), stride):
            wf = float(water[y0:y0 + size, x0:x0 + size].mean())
            if wf >= min_water_frac:
                rows.append((y0, x0, round(wf, 4)))
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def write_zarr_safe(ds, zpath, encoding):
    """Write a Zarr cube, tolerating NFS 'Directory not empty' on overwrite.

    Zarr v3 overwrites by shutil.rmtree'ing the old dir in place, which fails on
    networked filesystems when a chunk file is still open (silly-renamed to a
    hidden .nfs* file). So move any existing cube ASIDE (an atomic dir rename that
    works even with open handles), write fresh, then best-effort delete the stash.
    """
    zpath = Path(zpath)
    stash = None
    if zpath.exists():
        stash = zpath.with_name(f"{zpath.name}.old-{os.getpid()}-{int(time.time())}")
        zpath.rename(stash)
    ds.to_zarr(zpath, mode="w-", consolidated=True, encoding=encoding)
    if stash is not None:
        try:
            shutil.rmtree(stash)
        except OSError as exc:                 # NFS .nfs* leftovers -> non-fatal
            log.warning("  wrote %s but could not remove old cube %s (%s); "
                        "delete it later (likely a process still had it open)",
                        zpath.name, stash.name, exc)


def run(cfg, only_aoi):
    asm = cfg.get("assembler", {})
    root = Path(cfg.get("project", {}).get("root", "."))
    out_dir = root / cfg["paths"]["data"] / cfg["paths"]["training"]
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = asm.get("chunks", {"time": 64, "y": 128, "x": 128})
    size = int(asm.get("tile_size", 128))
    stride = int(asm.get("tile_stride", size))
    min_wf = float(asm.get("min_water_frac", 0.05))
    overwrite = asm.get("overwrite", False)

    days = pd.date_range(cfg["time"]["start_date"], cfg["time"]["end_date"], freq="D")
    aois = cfg["aois"]
    if only_aoi:
        req = set(only_aoi)
        aois = [a for a in aois if a["id"] in req]
        missing = req - {a["id"] for a in aois}
        if missing:
            raise SystemExit(f"AOI(s) not found in config: {sorted(missing)}")
        log.info("Assembling subset: %s", [a["id"] for a in aois])

    stats = StatsAccumulator(CONTINUOUS)
    tiles = []
    for aoi in aois:
        aid = aoi["id"]
        zpath = out_dir / f"{aid}.zarr"
        if zpath.exists() and not overwrite:
            log.info("=== %s: %s exists, skipping (use overwrite) ===", aid, zpath.name)
            ds = xr.open_zarr(zpath)
        else:
            log.info("=== assembling %s (%d days) ===", aid, len(days))
            ds = assemble_aoi(aoi, cfg, days)
            write_zarr_safe(ds, zpath, zarr_encoding(ds, chunks))
            log.info("  wrote %s  vars=%d shape=(t=%d,y=%d,x=%d)",
                     zpath.name, len(ds.data_vars), ds.sizes["time"],
                     ds.sizes["y"], ds.sizes["x"])
        stats.update(ds)
        for (y0, x0, wf) in tile_origins(ds, size, stride, min_wf):
            tiles.append({"aoi": aid, "y0": y0, "x0": x0, "size": size, "water_frac": wf})

    # global outputs
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(stats.result(), f, indent=2)
    pd.DataFrame(tiles).to_csv(out_dir / "tile_index.csv", index=False)
    log.info("Wrote %s and %s (%d tiles).",
             out_dir / "norm_stats.json", out_dir / "tile_index.csv", len(tiles))
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR datacube assembler.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", nargs="+", help="Process only these AOI id(s), space-separated.")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing .zarr cubes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.overwrite:
        cfg.setdefault("assembler", {})["overwrite"] = True
    run(cfg, args.aoi)


if __name__ == "__main__":
    main()