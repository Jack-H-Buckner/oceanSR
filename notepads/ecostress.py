import matplotlib.pyplot as plt
import xarray as xr

# Load the NetCDF file
ds = xr.open_dataset('data/ECOSTRESS/aligned/bellingham_bay/bellingham_bay_20230723T023254.nc')
print(ds)


# plt.figure(figsize=(10, 6))
# ds["sst"].plot(cmap="viridis")
# plt.savefig("results/bellingham_bay_sst_eco.png")

# plt.figure(figsize=(10, 6))
# ds["cloud"].plot(cmap="viridis")
# plt.savefig("results/eda/bellingham_bay_cloud_eco.png")

# plt.figure(figsize=(10, 6))
# ds["water"].plot(cmap="viridis")
# plt.savefig("results/eda/bellingham_bay_water_eco.png")

# plt.figure(figsize=(10, 6))
# ds["valid"].plot(cmap="viridis")
# plt.savefig("results/eda/bellingham_bay_valid_eco.png")