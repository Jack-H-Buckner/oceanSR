import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/MUR/aligned/cape_fear_estuary/cape_fear_estuary_20230104.nc')

print(ds)

plt.figure(figsize=(10, 6))
ds["sst"].plot(cmap="viridis")
plt.savefig("results/eda/cape_fear_sst_mur.png")

ds["valid"].plot(cmap="viridis")
plt.savefig("results/eda/cape_fear_valid_mur.png")