"""
This script gives an overview of some of validation tools available when
using the TidegaugeMultiple object in COAsT.

For this a script, a premade netcdf file containing tide gauge data is used.
"""
#%% 1. Import necessary libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import coast
import datetime

#%% 2. Define paths
fn_dom = "/Users/dbyrne/Projects/coast/workshops/07092021/data/mesh_mask.nc"
fn_dat = "/Users/dbyrne/Projects/coast/workshops/07092021/data/sossheig*"
fn_tg = "/Users/dbyrne/Projects/coast/workshops/07092021/data/tg_amm15.nc"

#%% 3. Create gridded object and load data
nemo = coast.Gridded(fn_dat, fn_dom, multiple=True, config="./config/example_nemo_grid_t.json")

# Create a landmask array and put it into the nemo object.
# Here, using the bottom_level == 0 variable from the domain file is enough.
nemo.dataset["landmask"] = nemo.dataset.bottom_level == 0

# Rename depth_0 to be depth
nemo.dataset = nemo.dataset.rename({"depth_0": "depth"})
nemo.dataset = nemo.dataset[["ssh", "landmask"]]


#%% 4. Create TidegaugeMultiple object

# Create the object and then inset the netcdf dataset
obs = coast.TidegaugeMultiple()
obs.dataset = xr.open_dataset(fn_tg)

# Cut down data to be only in 2018 to match model data.
start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime(2018, 12, 31)
obs.dataset = coast.general_utils.data_array_time_slice(obs.dataset, start_date, end_date)


#%% 5. Interpolate model data onto obs locations
model_timeseries = obs.obs_operator(nemo)

# In this case, transpose the interpolated dataset
model_timeseries.dataset = model_timeseries.dataset.transpose()


#%% 6. Do some processing
# This routine searches for missing values in each dataset and applies them
# equally to each corresponding dataset
obs, model_timeseries = obs.match_missing_values(model_timeseries)

# Subtract means from all time series
obs = obs.demean_timeseries()
model_timeseries = model_timeseries.demean_timeseries()

# Now you have equivalent and comparable sets of time series that can be
# easily compared.

#%% Calculate non tidal residuals

# First, do a harmonic analysis. This routine uses utide
ha_mod = model_timeseries.harmonic_analysis_utide()
ha_obs = obs.harmonic_analysis_utide()

# Create new TidegaugeMultiple objects containign reconstructed tides
tide_mod = model_timeseries.reconstruct_tide_utide(ha_mod)
tide_obs = obs.reconstruct_tide_utide(ha_obs)

# Get new TidegaugeMultiple objects containing non tidal residuals.
ntr_mod = model_timeseries.calculate_residuals(tide_mod)
ntr_obs = obs.calculate_residuals(tide_obs)

# Other interesting applications here included only reconstructing specified
# tidal frequency bands and validating this.

#%% Calculate errors

# The difference() routine will calculate differences, absolute_differences
# and squared differenced for all variables:
ntr_diff = ntr_obs.difference(ntr_mod)
ssh_diff = obs.difference(model_timeseries)

# We can then easily get mean errors, MAE and MSE
mean_stats = ntr_diff.dataset.mean(dim="t_dim", skipna=True)

#%% Threshold Statistics for Non-tidal residuals

# This is a simple extreme value analysis of whatever data you use.
# It will count the number of peaks and the total time spent over each
# threshold provided. It will also count the numbers of daily and monthly
# maxima over each threshold

thresh_mod = ntr_mod.threshold_statistics(thresholds=np.arange(0, 2, 0.2))
thresh_obs = ntr_obs.threshold_statistics(thresholds=np.arange(0, 2, 0.2))
