#!/usr/bin/env python3
"""
OCEANSR -- tide-height forcing (NOAA CO-OPS harmonic constituents).

Reads the common project config (configs/config.yaml): AOIs and dates are shared;
settings come from `sources.tides`. Tide is ~spatially uniform over a small AOI,
so for each AOI this finds the nearest CO-OPS water-level station, fetches its
published HARMONIC CONSTITUENTS (harcon.json -- one tiny, fast metadata request),
and computes the tide series LOCALLY with pytides2. This avoids the slow/flaky
`datagetter` prediction endpoint entirely and works for any date range.

    data/TIDE/aligned/<aoi_id>/<aoi_id>_tides.nc   (dims: time; var: tide [m], rel. MSL)

The datacube assembler broadcasts this across the AOI grid and samples it at the
daily / overpass times. Needs `requests` + `pytides2`.

Usage (run from the project root):
    python src/acquire_tides.py --config configs/config.yaml
    python src/acquire_tides.py --config configs/config.yaml --aoi hood_canal --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from shapely.geometry import box, shape
from shapely.ops import unary_union

log = logging.getLogger("acquire_tides")
SOURCE = "tides"


def _patch_legacy_compat():
    """pytides2 0.0.5 predates Py3.10 / NumPy>=1.24. Restore the aliases it uses
    (collections.Iterable etc., np.float etc.) so it imports cleanly."""
    import collections
    import collections.abc as _abc
    for _n in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable", "Hashable"):
        if not hasattr(collections, _n):
            setattr(collections, _n, getattr(_abc, _n))
    for _n, _t in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if not hasattr(np, _n):
            setattr(np, _n, _t)


_patch_legacy_compat()
STATIONS_MD = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
HARCON_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/harcon.json"


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
    root = Path(cfg.get("project", {}).get("root", "."))
    paths = cfg["paths"]
    return {
        "aois": cfg["aois"],
        "time": time_cfg,
        "ds": src_cfg,
        "out_dir": root / paths["data"] / paths[source] / "aligned",
        "fmt": src_cfg.get("output_format", "netcdf"),
        "overwrite": src_cfg.get("overwrite", False),
    }


# --------------------------------------------------------------------------- #
# Geometry + station selection
# --------------------------------------------------------------------------- #
def aoi_centroid_lonlat(aoi):
    if aoi.get("geometry"):
        import json
        gj = json.load(open(aoi["geometry"]))
        if gj.get("type") == "FeatureCollection":
            g = unary_union([shape(f["geometry"]) for f in gj["features"]])
        elif gj.get("type") == "Feature":
            g = shape(gj["geometry"])
        else:
            g = shape(gj)
    else:
        w, s, e, n = aoi["bbox"]
        g = box(w, s, e, n)
    c = g.centroid
    return c.x, c.y


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_stations() -> list:
    js = _get_json(STATIONS_MD, {"type": "waterlevels"})
    out = []
    for s in js.get("stations", []):
        try:
            out.append({"id": s["id"], "name": s["name"],
                        "lat": float(s["lat"]), "lon": float(s["lng"])})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def nearest_station(lon, lat, stations):
    best, best_d = None, float("inf")
    for s in stations:
        d = haversine_km(lon, lat, s["lon"], s["lat"])
        if d < best_d:
            best, best_d = s, d
    return best, best_d


# --------------------------------------------------------------------------- #
# CO-OPS metadata fetch + local harmonic prediction
# --------------------------------------------------------------------------- #
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "OCEANSR/1.0 (research; coastal SST pipeline)",
    "Accept": "application/json",
})


def _get_json(url, params=None, retries=5):
    """GET with a real UA + exponential backoff (metadata endpoints are fast)."""
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params or {}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            wait = min(30, 5 * (2 ** attempt))  # 5,10,20,30,30
            log.warning("    request failed (%s); retry %d/%d in %ds",
                        str(exc)[:70], attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"CO-OPS request failed after {retries} attempts: {last}")


def fetch_harcon(station_id) -> list:
    """Published harmonic constituents for a station (small, fast metadata call)."""
    js = _get_json(HARCON_URL.format(sid=station_id), {"units": "metric"})
    cons = js.get("HarmonicConstituents") or []
    if not cons:
        raise RuntimeError("no harmonic constituents returned")
    return cons


def predict_series(harcon, start, end, interval="h") -> pd.Series:
    """Compute the tide series locally from constituents (pytides2). Heights in m,
    relative to mean sea level (Z0 = 0). Nodal corrections handled by pytides2."""
    try:
        try:
            from pytides2.tide import Tide
            from pytides2.constituent import noaa
        except ImportError:  # original package name fallback
            from pytides.tide import Tide
            from pytides.constituent import noaa
    except ImportError as exc:
        raise RuntimeError(
            "pytides2 is not installed. Install it (deps come from conda) with:\n"
            "  mamba install -c conda-forge numpy scipy\n"
            "  pip install --no-build-isolation --no-deps pytides2"
        ) from exc

    # Plain Python lists, not numpy arrays: pytides2 does `None in [amps, phases]`
    # which raises "ambiguous truth value" if these are arrays.
    amps = [0.0] * len(noaa)
    phases = [0.0] * len(noaa)
    used = 0
    for hc in harcon:
        i = int(hc.get("number", 0)) - 1   # harcon 'number' indexes the NOAA order
        if 0 <= i < len(noaa):
            amps[i] = float(hc["amplitude"])
            phases[i] = float(hc["phase_GMT"])
            used += 1
    if used == 0:
        raise RuntimeError("constituents did not map to the NOAA set")

    tide = Tide(constituents=list(noaa), amplitudes=amps, phases=phases)
    freq = {"h": "h", "hourly": "h"}.get(str(interval), str(interval))
    times = pd.date_range(pd.Timestamp(start),
                          pd.Timestamp(end) + pd.Timedelta(days=1),
                          freq=freq, inclusive="left")
    heights = tide.at(list(times.to_pydatetime()))
    return pd.Series(np.asarray(heights, dtype="float32"), index=times, name="tide")


def write_output(ds, out_dir, aoi_id, fmt) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt != "netcdf":
        # Tide is 1D; geotiff doesn't apply -> always NetCDF.
        log.info("  (tide is a 1D series; writing NetCDF regardless of output_format)")
    path = out_dir / f"{aoi_id}_tides.nc"
    ds.to_netcdf(path, encoding={"tide": {"zlib": True, "complevel": 4}})
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(eff, only_aoi, dry_run):
    ds_cfg = eff["ds"]
    out_root, fmt, overwrite = eff["out_dir"], eff["fmt"], eff["overwrite"]
    start, end = eff["time"]["start_date"], eff["time"]["end_date"]

    aois = eff["aois"]
    if only_aoi:
        aois = [a for a in aois if a["id"] == only_aoi]
        if not aois:
            raise SystemExit(f"AOI '{only_aoi}' not found in config.")

    stations = None  # lazy-load station metadata only if needed
    for aoi in aois:
        aoi_id = aoi["id"]
        lon, lat = aoi_centroid_lonlat(aoi)

        if aoi.get("tide_station"):
            station = {"id": str(aoi["tide_station"]), "name": "(config override)",
                       "lat": lat, "lon": lon}
            dist = 0.0
        else:
            if stations is None:
                log.info("Fetching CO-OPS water-level station list...")
                stations = fetch_stations()
                log.info("  %d stations", len(stations))
            station, dist = nearest_station(lon, lat, stations)

        log.info("=== AOI: %s (%s) | station %s '%s' (%.1f km) ===",
                 aoi_id, aoi.get("name", ""), station["id"], station["name"], dist)
        if dist > 75:
            log.warning("    nearest gauge is %.0f km away -- consider a tide_station override", dist)

        out_path = out_root / aoi_id / f"{aoi_id}_tides.nc"
        if not overwrite and out_path.exists():
            log.info("  already processed, skipping")
            continue
        if dry_run:
            log.info("  [dry-run] would fetch harcon for %s and predict %s..%s @ %s",
                     station["id"], start, end, ds_cfg.get("interval", "h"))
            continue

        try:
            harcon = fetch_harcon(station["id"])
            s = predict_series(harcon, start, end, ds_cfg.get("interval", "h"))
        except Exception as exc:
            log.warning("  skipping %s (%s)", aoi_id, exc)
            continue

        da = xr.DataArray(s.values.astype("float32"),
                          coords={"time": s.index.values}, dims="time", name="tide")
        da.attrs.update(units="m", long_name="tide height (harmonic prediction, rel. MSL)")
        ds = da.to_dataset()
        ds.attrs.update(aoi_id=aoi_id, station_id=station["id"], station_name=station["name"],
                        station_lat=station["lat"], station_lon=station["lon"],
                        distance_km=round(dist, 2), n_constituents=len(harcon),
                        method="harmonic synthesis (pytides2)", datum="MSL")
        log.info("  wrote %s (%d steps)", write_output(ds, out_root / aoi_id, aoi_id, fmt),
                 ds.sizes["time"])
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="OCEANSR tide-height acquisition (NOAA CO-OPS).")
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