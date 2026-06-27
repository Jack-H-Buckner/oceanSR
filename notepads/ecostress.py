import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/ECOSTRESS/aligned/padilla_bay/padilla_bay_20230101T105856.nc')

plt.figure(figsize=(10, 6))
ds["sst"].plot(cmap="viridis")
plt.savefig("results/padilla_bay_sst.png")

plt.figure(figsize=(10, 6))
ds["cloud"].plot(cmap="viridis")
plt.savefig("results/eda/padilla_bay_cloud.png")

plt.figure(figsize=(10, 6))
ds["water"].plot(cmap="viridis")
plt.savefig("results/eda/padilla_bay_water.png")

plt.figure(figsize=(10, 6))
ds["valid"].plot(cmap="viridis")
plt.savefig("results/eda/padilla_bay_valid.png")