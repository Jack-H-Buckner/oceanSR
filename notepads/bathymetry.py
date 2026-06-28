import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/BATHYMETRY/aligned/padilla_bay/padilla_bay.nc')

print(ds)

plt.figure(figsize=(10, 6))
ds["depth"].plot(cmap="viridis")
plt.savefig("results/padilla_bay_depth.png")

plt.figure(figsize=(10, 6))
ds["elevation"].plot(cmap="viridis")
plt.savefig("results/padilla_bay_elevation.png")