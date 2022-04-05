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

fn_dom = dir + "<PATH_TO_NEMO_DOMAIN"
fn_dat = dir + "<PATH_TO_NEMO_DATA"
fn_tg = dir + "<PATH_TO_TIDEGAUGE_NETCDF"  # This should already be processed, on the same time dimension

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
obs = coast.Tidegauge(dataset=xr.open_dataset(fn_tg))

# Cut down data to be only in 2018 to match model data.
start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime(2018, 12, 31)
obs = obs.time_slice(start_date, end_date)

#%% 5. Interpolate model data onto obs locations
model_timeseries = obs.obs_operator(nemo)

# In this case, transpose the interpolated dataset
model_timeseries.dataset = model_timeseries.dataset.transpose()


#%% 6. Do some processing
# This routine searches for missing values in each dataset and applies them
# equally to each corresponding dataset
tganalysis = coast.TidegaugeAnalysis()
obs_new, model_new = tganalysis.match_missing_values(obs.dataset.ssh, model_timeseries.dataset.ssh)

# Subtract means from all time series
obs_new = tganalysis.demean_timeseries(obs_new.dataset)
model_new = tganalysis.demean_timeseries(model_new.dataset)

# Now you have equivalent and comparable sets of time series that can be
# easily compared.

#%% Calculate non tidal residuals

# First, do a harmonic analysis. This routine uses utide
ha_mod = tganalysis.harmonic_analysis_utide(model_new.dataset.ssh)
ha_obs = tganalysis.harmonic_analysis_utide(obs_new.dataset.ssh)

# Create new TidegaugeMultiple objects containing reconstructed tides
tide_mod = tganalysis.reconstruct_tide_utide(model_new.dataset.time, ha_mod)
tide_obs = tganalysis.reconstruct_tide_utide(obs_new.dataset.time, ha_obs)

# Get new TidegaugeMultiple objects containing non tidal residuals.
ntr_mod = tganalysis.calculate_residuals(model_new.dataset.ssh, tide_mod.dataset.ssh)
ntr_obs = tganalysis.calculate_residuals(obs_new.dataset.ssh, tide_obs.dataset.ssh)

# Other interesting applications here included only reconstructing specified
# tidal frequency bands and validating this.

#%% Calculate errors

# The difference() routine will calculate differences, absolute_differences
# and squared differenced for all variables:
ntr_diff = tganalysis.difference(ntr_obs, ntr_mod)
ssh_diff = tganalysis.difference(obs_new, model_new)

# We can then easily get mean errors, MAE and MSE
mean_stats = ntr_diff.dataset.mean(dim="t_dim", skipna=True)

#%% Threshold Statistics for Non-tidal residuals

# This is a simple extreme value analysis of whatever data you use.
# It will count the number of peaks and the total time spent over each
# threshold provided. It will also count the numbers of daily and monthly
# maxima over each threshold

thresh_mod = tganalysis.threshold_statistics(ntr_mod, thresholds=np.arange(0, 2, 0.2))
thresh_obs = tganalysis.threshold_statistics(ntr_obs, thresholds=np.arange(0, 2, 0.2))
