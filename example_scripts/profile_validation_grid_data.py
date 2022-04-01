import coast
import numpy as np
import xarray as xr

fn_in = "/Users/dbyrne/transfer/surface_data_test.nc"
fn_out = "/Users/dbyrne/transfer/surface_data_gridded_test.nc"

#%%

# Load in data for averaging (e.g. surface data)
prof_data = coast.Profile(fn_in)

# Define longitude and latitude grid
grid_lon = np.arange(-15, 15, 0.5)
grid_lat = np.arange(45, 65, 0.5)

# Average all data across all seasons
prof_gridded = prof_data.average_into_grid_boxes(grid_lon, grid_lat, min_datapoints=0)

# Average data for each season
prof_gridded_DJF = prof_data.average_into_grid_boxes(grid_lon, grid_lat, season="DJF", var_modifier="_DJF")
prof_gridded_MAM = prof_data.average_into_grid_boxes(grid_lon, grid_lat, season="MAM", var_modifier="_MAM")
prof_gridded_JJA = prof_data.average_into_grid_boxes(grid_lon, grid_lat, season="JJA", var_modifier="_JJA")
prof_gridded_SON = prof_data.average_into_grid_boxes(grid_lon, grid_lat, season="SON", var_modifier="_SON")

# Merge together
ds_prof_gridded = xr.merge(
    (
        prof_gridded.dataset,
        prof_gridded_DJF.dataset,
        prof_gridded_MAM.dataset,
        prof_gridded_JJA.dataset,
        prof_gridded_SON.dataset,
    )
)

# Save to file
ds_prof_gridded.to_netcdf(fn_out)
