import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/MET/aligned/tillamook_bay/tillamook_bay_20230106.nc')

print(ds)