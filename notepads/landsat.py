import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/LANDSAT/aligned/hood_canal/hood_canal_20230103T190217.nc')

# Inspect dimensions, coordinates, and variables
print(ds)
plt.figure(figsize=(10, 6))
ds["sst"].plot(cmap="viridis")
plt.savefig("results/hood_canal_sst.png")

plt.figure(figsize=(10, 6))
ds["cloud"].plot(cmap="viridis")
plt.savefig("results/eda/hood_canal_cloud.png")

plt.figure(figsize=(10, 6))
ds["water"].plot(cmap="viridis")
plt.savefig("results/eda/hood_canal_water.png")

plt.figure(figsize=(10, 6))
ds["valid"].plot(cmap="viridis")
plt.savefig("results/eda/hood_canal_valid.png")
