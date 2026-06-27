# Ocean temeprature super resolution and interpolation

This project builds high resoltuion level-4 sea surface temeprature data products for coastal areas using a foundation modeling appraoch.  The model uses a 3d UNET archetecture to create embeddings of thermal remote sensing images. These data will then be fine tuned to predict bulk sea surface temeperatures using in situ bouy data. 

The model combines three primary data sets, Landsat and ECOSTRESS thermal images for high resolution values and MUR-SST level-4 data products for a continuously avaible benchmark. The Landsat and ECOSTRESS images will provide information on fine scale structures and near shore effects missing from the MUR-sst data product. 

Finally, the fine tuning with the in situ measurements will help correct for skin-bulk sst differnces and biases included in the SST retrival algorithms for each instrument. 

# Data sources

## ECOSTRESS
The primary set of high resoluton thermal images comes from the ISS ECOSTRESS isntrument these are loaded through NASA earth acess API. These data are loaded for areas of interest defined in configs/congif.yaml and interpolated to a common grid with a defual of 100m resolution. The ecostress images also have land and cloud mask channels that are interpolated to the same grid 

The data is loaded by the 