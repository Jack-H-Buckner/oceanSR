# Implementation Plan: Thermal SST Spatiotemporal Model (3D U-Net Benchmark)

A staged plan for a self-supervised spatiotemporal model that fuses high-resolution thermal imagery (ECOSTRESS, Landsat) with a gap-free coarse backbone (MUR L4) to produce continuous 100 m SST, then calibrates to bulk SST with sparse buoy observations. PyTorch. Target hardware: NVIDIA DGX Spark (128 GB unified, ~100 TFLOPS BF16). The 3D U-Net is the **benchmark** before attempting the spatiotemporal ViT.

---

## Design principles (carry through every stage)

1. **AOI-centric cubes, then tile.** Build a small datacube *per area of interest* (each fjord/estuary), never from full Landsat/ECOSTRESS granules. Train on 256×256 crops — or the whole AOI if it's smaller — × a short temporal window (T = 8–16 days). Domain size is a data-loading concern, not a model-size concern; clipping to AOIs makes storage and I/O an order of magnitude smaller.
2. **Common grid + mask channels.** Resolve all resolution/availability differences in the data representation. Every *gappy* source (thermal) = a value channel + a binary availability-mask channel; every *complete* driver (forcing) = an unmasked channel available at all pixels/times.
3. **Residual on MUR.** Output = `upsample(MUR) + residual`. The network only learns high-frequency detail, never the large-scale field from scratch.
4. **Learn radiometric bias, preprocess geometry.** Per-source affine offset (`a_s·x + b_s`) is a model parameter; reprojection/regridding and cloud-QA masking are still done on disk.
5. **Two stages.** (A) Self-supervised masked reconstruction/forecasting on the cube. (B) Sparse fine-tune to bulk SST on buoys, with the in-situ data supplying the absolute anchor.
6. **Complete forcing drives prediction, not just correction.** Air temp, wind, shortwave, tide, and time-of-day are gap-free, so they enter the **embedding/encoder** to help predict occluded days (forcing-informed interpolation), and are *also* routed to the skin→bulk correction head.

---

## Stage 0 — Environment & scope

- **Env:** Python 3.11, PyTorch (CUDA build for GB10/Blackwell), plus `earthaccess`, `xarray`, `rioxarray`, `rasterio`, `pyresample` or `xesmf`, `zarr`, `dask`, `numpy`, `pandas`, `netCDF4`. For forcing data: `herbie-data` + `cfgrib`/`pygrib` (NOAA HRRR/RTMA/NARR GRIB2) and `noaa_coops` (CO-OPS tides). Add `flash-attn` later for the ViT phase only.
- **Define an AOI registry.** A GeoJSON/shapefile of target water bodies — each fjord/estuary as a polygon plus a small shoreline buffer — with fields `aoi_id, name, geometry, local_CRS`. The entire pipeline (acquisition, cubes, forcing, in-situ, sampling) loops over this registry. Start with a handful of AOIs and ~1–2 years, expand later. (Sources for the polygons: NOAA estuary boundaries / NERRS reserves, USGS NHD waterbodies, or hand-drawn boxes.)
- **Per-AOI canonical grid.** For each AOI, define a *local* grid (local UTM zone, 100 m posting, daily step) snapped to the buffered AOI extent. Store each grid's (CRS, bounds, shape, transform) in config so acquisition, cube-building, training, and inference agree. **Train across all AOIs jointly** — the diversity of fjords/estuaries is what makes this a foundation model rather than a single-site fit.

---

## Stage 1 — Data acquisition

**Loop this entire stage over the AOI registry.** Search each source by the **AOI bounding box** and **clip/crop granules to the AOI polygon at ingest** — you never build cubes from full Landsat/ECOSTRESS granules. All NASA sources go through **Earthdata Login** + the `earthaccess` library (handles auth, search, cloud/HTTPS download).

**MUR L4 SST (backbone, gap-free, 0.01°/~1 km daily) — PODAAC/GHRSST.**
```python
import earthaccess
earthaccess.login()
res = earthaccess.search_data(
    short_name="MUR-JPL-L4-GLOB-v4.1",   # GHRSST MUR L4
    temporal=("2023-01-01", "2024-12-31"),
    bounding_box=(W, S, E, N),
)
earthaccess.download(res, "raw/mur/")
```
Variables: `analysed_sst` (Kelvin), plus `mask`/`sea_ice_fraction` if useful. Daily, globally complete → your always-present input. **Caveat for narrow AOIs:** at 1 km, MUR may not resolve a fjord/estuary only a few hundred metres wide — its pixels can be flagged land or contaminated by adjacent land. Where MUR is missing/unreliable inside the AOI, fall back to the nearest valid open-water MUR pixel as a regional backbone (and lean more on the high-res sources + forcing). Flag this per AOI; it's the main weak point of the "MUR fills gaps" premise in nearshore waters.

**ECOSTRESS LST (high-res detail, ~70 m, irregular ISS overpass) — LP DAAC.**
- Product: `ECO_L2T_LSTE` (tiled Level-2 land surface temperature; over-water pixels usable as skin SST). Search via `earthaccess` with `short_name="ECO_L2T_LSTE"`, same bbox/temporal. Keep the `LST`, `QC`, and `cloud` layers.
- Note overpass **time of day varies** (ISS precesses) — record each granule's acquisition timestamp; you'll feed overpass-hour as a channel.

**Landsat 8/9 Collection-2 Level-2 Surface Temperature (~100 m TIRS, 30 m grid, 16-day) — USGS.**
- Not on Earthdata; pull Collection-2 L2 ST via a STAC API (USGS Landsat STAC, Microsoft Planetary Computer, or AWS Open Data) or USGS M2M. Keep the `ST_B10` surface-temperature band and the `QA_PIXEL` cloud/cloud-shadow band.

**In-situ bulk SST (fine-tune labels) — per AOI.** For fjords/estuaries, the best US in-situ sources are often *not* offshore buoys:
- **NERRS SWMP** (National Estuarine Research Reserve System-Wide Monitoring Program) — fixed estuarine water-quality stations reporting water temperature; ideal where your AOIs overlap a reserve.
- **IOOS regional associations** — e.g., AOOS (Alaska fjords), NANOOS (Pacific NW), CeNCOOS — aggregate coastal/estuarine moorings and gliders.
- **NOAA NDBC** + **CO-OPS** station water temperature for the more open-water AOIs.
- Pull water-temperature time series for stations inside each AOI; keep station lat/lon, timestamp, value, measurement depth, and co-located wind if available. These are the sparse point truth for Stage B (leave-one-station-out validation later).

**Forcing / driver variables (gap-free) — US sources.** These are complete in space and time, so they feed the **embedding/encoder** to help predict *missing days* (not just correct skin→bulk). Regrid the gridded ones to the canonical grid by bilinear upsampling (smooth large-scale fields, so lossless enough); all carry **no availability mask** (or an all-ones mask).

- **Air temperature (2 m) + wind (10 m u/v) + downward shortwave** — NOAA **HRRR** (3 km, hourly, CONUS + coastal waters; on AWS Open Data) is the primary choice; retrieve with the `herbie-data` library. Alternatives: **RTMA** (2.5 km hourly surface *analysis*) for a clean analysis product, or **NARR** (32 km, 3-hourly) for a long, temporally consistent historical record. Keep `TMP:2 m`, `UGRD/VGRD:10 m`, and `DSWRF` (shortwave — the diurnal warm-layer driver).
- **Tide height / water level** — NOAA **CO-OPS Tides & Currents API** (`noaa_coops`): observed `water_level` plus harmonic `predictions` at coastal gauges. Tide is low-spatial-frequency over a small nearshore box, so broadcast/interpolate the nearest-station series across the tile (or use a regional NOAA **OFS** model for a true tidal field). Harmonic predictions are gap-free and extend to any date — an ideal complete driver.
- **Time of day & season** — not a download: derive cyclical encodings `sin/cos(2π·hour/24)` and `sin/cos(2π·doy/365)`, broadcast as channels. These give the model diurnal + seasonal phase for both gap-filling and forecasting.

**Static covariates.**
- Bathymetry (GEBCO, or NOAA **CRM**/coastal DEMs for US nearshore) and a land/water mask, regridded once to the canonical grid as static channels. (Coastline can come from the land mask.)

Deliverable of this stage: raw granules on disk + a manifest table (`source, path, datetime, bbox, qa_path`).

---

## Stage 2 — Harmonize into cubes + masks (the core preprocessing)

Goal: turn heterogeneous granules into **one aligned, chunked Zarr datacube per AOI** on that AOI's canonical 100 m / daily grid. Run this loop once per AOI.

**Per granule:**
1. **Clip to the AOI polygon** (buffered), then reproject + resample to that AOI's canonical grid. ECOSTRESS/Landsat → area-weighted/conservative resampling to 100 m; MUR (1 km) → bilinear upsample to 100 m (stays smooth, fine).
2. Apply QA: set cloud/cloud-shadow/low-quality pixels to NaN using the source's QA band.
3. Snap acquisition to its **daily time bin**.

**Land dominates the bounding box in fjords/estuaries**, so the land/water mask is load-bearing here, not cosmetic: carry it as a channel, exclude land pixels from all losses, and (Stage 3) weight tile sampling by water fraction so the model isn't trained mostly on land.

**Assemble the cube** with dimensions `(time, y, x)` per variable and write channels:

| Channel | Source | Notes |
|---|---|---|
| `mur` | MUR | gap-free backbone; reference frame |
| `eco`, `eco_mask` | ECOSTRESS | value (NaN→0 filled) + 1/0 availability |
| `lst`, `lst_mask` | Landsat | value + availability |
| `eco_hour`, `lst_hour` | metadata | overpass hour-of-day (diurnal cue) |
| `airtemp` | HRRR/RTMA | 2 m air temperature; complete, no mask |
| `wind_u`, `wind_v` | HRRR/RTMA | 10 m wind components; complete, no mask |
| `swrad` | HRRR | downward shortwave; diurnal driver |
| `tide` | CO-OPS / OFS | water-level/tide series, broadcast across tile |
| `t_hour`, `t_doy` | derived | sin/cos time-of-day + day-of-year |
| `bathy`, `landmask` | static | broadcast across time |

The forcing/driver channels (`airtemp`, `wind_*`, `swrad`, `tide`, `t_*`) are gap-free, so they get an all-ones mask (or none) — distinct from the thermal observation channels, whose masks flag real gaps.

Fill missing values with 0 (the mask, not the fill value, tells the model what's real). **Normalize per source** (store mean/std in config; apply consistently at train and inference). Convert all temperatures to a common unit (°C or K) up front.

**Zarr schema:** chunk as `(time: T_window, y: 256, x: 256)` so a training tile is ~one chunk read. Keep a sidecar table of valid (t, y, x) tile origins (e.g., tiles with ≥1 high-res observation in the window) for sampling.

> This stage is the only heavy preprocessing. It's geometric alignment + cloud masking — the radiometric cross-sensor harmonization is deliberately left to the model (Stage 4).

---

## Stage 3 — Dataset / dataloader

A `torch.utils.data.Dataset` that, per item, reads one space-time tile from Zarr and returns a dict:

```python
{
  "x":      FloatTensor[C, T, 256, 256],   # all value + mask + static + hour channels
  "valid":  FloatTensor[T, 256, 256],      # union availability of high-res sources (for loss)
  "mur":    FloatTensor[T, 256, 256],      # for the residual base
  "src_id": LongTensor[C_src],             # which source each value channel belongs to
}
```
- Sample tiles across **all AOI cubes** (one global valid-tile table tagged with `aoi_id`); weight toward windows richer in high-res observations **and higher water fraction**. For AOIs smaller than the tile, return the whole AOI (zero-padded) instead of cropping.
- Optionally pass an `aoi_id` embedding (or static AOI descriptors — mean depth, fjord-vs-estuary class) so the model can specialize per site while sharing weights; start without it and add only if cross-AOI bias appears.
- Augmentations that preserve geophysics: random flips/90° rotations (translation/rotation are safe for fields); avoid intensity jitter that would corrupt absolute temperature.
- Use `num_workers`, `pin_memory`, and Zarr/dask prefetch — at 273 GB/s, I/O is the likely bottleneck, so stage data fast.

---

## Stage 4 — Architecture (3D U-Net)

Standard encoder–decoder 3D U-Net with skip connections, plus three task-specific pieces.

**Backbone.**
- Input `(B, C, T, H, W)`. 3–4 downsampling levels, `Conv3d → GroupNorm → SiLU` blocks, base width 32–48 channels doubling per level. Downsample spatially (and optionally temporally) with strided conv; upsample with trilinear + conv. Skip connections per level. ~5–30 M params — trains in hours.

**(a) Per-source learned affine offset.** A parameter table `affine[n_sources, 2]`, applied to each value channel as `a_s·x + b_s` at input. **Freeze MUR's row at (1, 0)** as the reference frame; init others at (1, 0); little/no weight decay. This absorbs static radiometric bias with no preprocessing. (Upgrade later: make `b_s` a tiny MLP of overpass-hour for a diurnal-conditional offset.)

**(b) Residual-on-MUR output head.** Final `Conv3d` → residual; `prediction = upsample_to_100m(MUR) + residual`. Optionally bound the residual (e.g., `tanh`×range) for stability early in training.

**(c) Mask channels** are already in `x`; the network reads availability directly. (A later upgrade is partial/gated convolutions that renormalize by valid fraction; not needed for the benchmark.)

**(d) Driver channels enter at the embedding/stem.** The complete forcing variables (`airtemp`, `wind_*`, `swrad`, `tide`, `t_hour`, `t_doy`) are concatenated into the input so the first conv (the "embedding") sees them at every pixel and time step. This is what lets the model *predict occluded days* — propagating SST through cloud gaps via wind advection, air-temperature/insolation heating, and tidal mixing, rather than pure spatial smoothing. Because they're complete, they're equally available on masked/forecast frames, so they carry real predictive signal exactly where the thermal channels are missing.

**(e) Skin→bulk correction head.** Keep the backbone predicting the *skin* field; add a small **pointwise (1×1 conv) head** that outputs the skin→bulk delta from local forcing: `bulk = skin + Δ(skin, airtemp − skin, |wind|, swrad, tide, hour)`. Pointwise matches the local column physics (a data-driven warm-layer/cool-skin scheme) and lets a buoy-learned correction generalize across space; regularize Δ toward small magnitude. The *same* driver channels thus serve double duty — prediction at the embedding (d) and correction here (e). Supervised by buoys in Stage 6; the skin field stays anchored by the thermal reconstruction.

Output: dense skin field + bulk field `(B, T, H, W)` (or a single target frame — see Stage 5/6).

---

## Stage 5 — Self-supervised pretraining

**Objective:** reconstruct held-out high-res pixels from the rest of the cube. Mix masking patterns so the model learns the real gap structure:
- **Random tubelet masking** (interpolation) — hide blocks of high-res pixels.
- **Last-frame masking** (forecasting) — hide the final time step's high-res data.
- **Cloud-shaped masking** — overlay realistic cloud silhouettes (sample from actual cloud masks) so masked regions look like true occlusion.

Because the driver channels (wind, air temp, shortwave, tide, time) stay complete on masked/forecast frames, gap-filling becomes **forcing-informed prediction**: the model learns to use physical forcing to fill occluded days, not just spatially smooth. This is the main payoff of adding them at the embedding.

**Loss — mask-weighted, high-res-targeted.** Supervise only where a high-res observation exists **and** was masked out; do **not** weight MUR pixels as targets (MUR is a smoothed prior, not truth):
```
L = Σ_{masked & observed} w · (pred − target)² / N
```
Use Huber/L1 for robustness to thermal outliers. Add a light smoothness/TV term on the residual if fields look noisy.

**Training config (DGX Spark):** BF16 autocast, gradient checkpointing on, AdamW, cosine schedule + warmup, batch size as large as 128 GB allows (start 16–32 tiles), gradient accumulation if needed. Expect convergence in **hours to ~1 day** for the U-Net — well inside the weekend, leaving margin for sweeps. Log reconstruction RMSE on held-out masked pixels; checkpoint best.

---

## Stage 6 — Sparse fine-tuning on buoy bulk SST

Now adapt the skin-SST field to **bulk** SST using point observations.

**Match-ups.** For each buoy reading, find the model tile/time covering its (lat, lon, day). Build a table `(t, y, x, bulk_value, station_id, wind, hour)`.

**Sparse loss via differentiable sampling.** Run the field forward, then **bilinearly sample** the predicted field at each buoy's exact sub-pixel location (`grid_sample`), compare to bulk:
```
L_point = Σ_i ( field_pred(x_i, y_i, t_i) − bulk_i )² / N_obs
```
Gradients flow only through sampled points but update shared weights — the conv's translation-equivariance spreads the correction across all space.

**Keep a dual objective.** Train on `L_point + λ·L_recon` so the field stays physically coherent away from buoys (point-only supervision lets the field drift). Ramp `λ_point` up gradually.

**Guard against overfitting the tiny label set.**
- **Freeze the backbone** (or very low LR); train mainly the pointwise skin→bulk head from Stage 4(e), fed the buoy-pixel forcing (air temp, `airtemp − skin`, wind speed, shortwave, tide, overpass-hour) + bathymetry → bulk SST. This is where the diurnal warm-layer / cool-skin correction is learned. Sample the forcing at the **observation's hour**, not a daily mean (the warm layer is instantaneous), and include buoy measurement depth as a covariate.
- The buoy anchor now **pins the absolute reference**, resolving the offset degeneracy left from pretraining.

**Validation — leave-one-station-out.** Hold out *entire buoys*, never random points; random splits leak spatial structure and overstate generalization. Report per-station RMSE/bias.

---

## Stage 7 — Validation, budget & escalation

- **Quantitative:** masked reconstruction RMSE (Stage 5); leave-one-station-out *and* leave-one-AOI-out bulk-SST RMSE/bias (Stage 6); compare against two baselines — bilinear-upsampled MUR, and classical DINEOF gap-filling — to prove the model earns its complexity.
- **Qualitative:** inspect filled fields across cloud gaps and coastlines for artifacts (ringing, seams at tile edges); use overlapping-tile inference with feathering at deployment.
- **Compute sanity:** U-Net @ 256² × T=12, ~10–30 M params, BF16 + checkpointing → comfortably <128 GB at batch 32; ~hours/epoch. If I/O-bound, pre-stage Zarr to local NVMe and increase workers.
- **Escalation path:** once this benchmark's numbers are trusted, swap the backbone for the spatiotemporal MAE-ViT (Prithvi-style 3D tubelets, high masking) reusing the *same* cube, masks, offsets, residual head, and losses — only the encoder changes.

---

## Key risks / decisions to revisit

- **Daily binning discards ECOSTRESS diurnal sampling.** Acceptable for the benchmark; revisit with a sub-daily grid + GOES if diurnal SST matters.
- **Buoy scarcity** caps fine-tune signal — consider supplementing with drifters/Argo or in-situ-calibrated L4 products if leave-one-station-out is unstable.
- **Inter-sensor bias may be spatial/scene-varying**, which a single scalar offset can't capture; escalate to per-scene or low-rank offsets if residuals look structured.
- **MUR is itself a blended product** (partly built from the same sensors) — watch for circularity; keep it strictly as input/prior, never as the reconstruction target.
- **1 km backbone vs. sub-km AOIs.** MUR may not resolve narrow fjords/estuaries; the backbone can be weak or absent exactly where you care most. Mitigate with nearest-open-water fallback and heavier reliance on high-res + forcing; consider a higher-resolution regional L4 if one exists for your coast.
- **AOI generalization.** Training across many fjords/estuaries should help, but a model fit to a few sites may not transfer to a new AOI. Validate with **leave-one-AOI-out** in addition to leave-one-station-out, and keep the per-AOI grids consistent (same resolution, projection convention) so geometry isn't a confound.
- **Land contamination in narrow waters.** Mixed land/water pixels and shoreline thermal bleed are worse in estuaries; be strict with the water mask and consider eroding the shoreline by a pixel before computing losses.