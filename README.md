# Ocean Temperature Super Resolution and Interpolation

This project builds high resolution level-4 sea surface temperature data products for coastal areas using a foundation modeling approach. The model uses a 3D UNET architecture to create embeddings of thermal remote sensing images. These embeddings are then fine-tuned to predict bulk sea surface temperatures using in situ buoy data.

The model combines three primary data sets: Landsat and ECOSTRESS thermal images for high resolution values, and MUR-SST level-4 data products for a continuously available benchmark. The Landsat and ECOSTRESS images provide information on fine scale structures and near-shore effects missing from the MUR-SST data product.

Finally, fine-tuning with in situ measurements helps correct for skin-bulk SST differences and biases included in the SST retrieval algorithms for each instrument.

```
OCEANSR/
├── configs/
│   └── config.yml          # shared AOIs / grid / dates / paths + per-source blocks
├── data/
│   ├── ECOSTRESS/          # raw/ (cached COGs) + aligned/ (per-overpass NetCDF)
│   ├── LANDSAT/
│   ├── TRAINING/
│   └── ...
├── results/
├── src/
│   └── acquire_ecostress.py # Stage 1a: ECOSTRESS V3 SST acquisition
├── requirements.txt
└── README.md
```

## Data sources

### ECOSTRESS

The primary set of high resolution thermal images comes from the ISS ECOSTRESS instrument. These are loaded through the NASA earthaccess API for areas of interest defined in `configs/config.yml`, and interpolated to a common grid with a default of 100 m resolution. The ECOSTRESS images also have land and cloud mask channels that are interpolated to the same grid.

The data is loaded by the `acquire_ecostress.py` file for the date ranges and areas of interest listed in `config.yml`.

### Landsat

Landsat provides a secondary source of high resolution thermal images for the model. These are loaded through Google Earth Engine and interpolated to a 100 m grid. The Landsat images have a cloud channel that is used to mask pixels likely to be contaminated.

The data is loaded by the `acquire_landsat.py` file for the date ranges and areas of interest listed in `config.yml`.

### MUR SST

We use the MUR SST data product as the gap-free backbone for the SST models. MUR SST is a NASA data product that merges sea surface temperature estimates from several sensors into one gap-free field. Despite its advantages, the MUR data product is often biased in nearshore areas. Our models are designed to learn this bias.

This data is loaded through NASA earthaccess by the `acquire_mur.py` file.

### Meteorological data

The thermal images we use to predict sea surface temperatures measure skin temperatures, which can differ from bulk sea surface temperatures. Wind and air temperature are the primary factors that influence this difference. To account for these factors, we include meteorological reanalysis data in the sea surface temperature models from the NOAA HRRR 3 km model. We also load cloud cover data from this source to identify scenes with clear skies for training.

This data is loaded through NASA earthaccess by the `acquire_met.py` file.

### Bathymetry data

Depth and local bathymetry can influence sea surface temperatures. We use two sources for bathymetry maps: high resolution maps from the NOAA CUDEM model, and low resolution bathymetry from the global GMRT model as a fallback.

This data is loaded through NASA earthaccess by the `acquire_bathymetry.py` file.

### Tides

Tidal forcing can influence changes in sea surface temperature between days. We use a tide model that predicts tide heights using the NOAA CO-OPS harmonic constituents, loaded and processed through the `acquire_tides.py` file.

## Data packaging

The `assemble_datacube.py` file combines all of the data sources for an area of interest (AOI) into a `.zarr` format. All variables are mapped to a 100 m resolution daily grid and saved to the `data/TRAINING` folder.