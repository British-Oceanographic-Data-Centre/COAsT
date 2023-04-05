# This script demonstrates how to use the Profile and Gridded objects to
# extract model profiles and do some comparisons with observed profiles.
# It will do a nearest neighbour extraction of model data (with time interpolation
# of your choice). It will then calculate differences between the model
# and obs and averaged profiles and errors into surface and bottom layers.
#
#   This script will result in five new files being written:
#
#   1. extracted_profiles: Model data on model levels extracted at obs locs
#   2. interpolated_profiles: Model data on ref depth level
#   3. interpolated_obs: Obs data on ref depth levels
#   4. profile_errors: Differences between interpolated_profiles and _obs
#   5. surface_data: Surface data and errors
#   6. bottom_data: Bottom data and errors
#
# If you are dealing with very large datasets, you should take a look at the
# script analysis_extract_and_compare_single_process.py. This script demonstrates
# a single process that can be used to build a parallel scheme.
#
# This script can be used with COAsT example data. Please set:
#
# fn_dom = path.join('./example_files', "coast_example_nemo_domain.nc")
# fn_dat = path.join('./example_files', "coast_example_nemo_data.nc")
# dn_out = "./example_files"
# fn_prof = path.join('./example_files', "coast_example_EN4_201008.nc")
# fn_cfg_nemo = path.join('./config', "example_nemo_grid_t.json")
# fn_cfg_prof = path.join('./config', "example_en4_profiles.json")

import sys

# IF USING A DEVELOPMENT BRANCH OF COAST, ADD THE REPOSITORY TO PATH:
# sys.path.append('<PATH_TO_COAST_REPO')
import coast
import xarray as xr
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import os.path as path

print("Modules loaded", flush=True)

# Name of the run -- used mainly for naming output files
run_name = "co7"

# Figure out what the date range is for this analysis process
start_date = datetime.datetime(2007, 1, 1)
end_date = datetime.datetime(2010, 12, 1)
print("Analysis Range: {0} -->> {1}".format(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")), flush=True)

# Depth averaging settings
ref_depth = np.concatenate((np.arange(1, 100, 2), np.arange(100, 300, 5), np.arange(300, 1000, 50)))
surface_def = 5  # in metres
bottom_height = [10, 30, 100]  # Use bottom heights of 10m, 30m and 100m for...
bottom_thresh = [100, 500, np.inf]  # ...depths less than 100m, 500m and infinite

# File paths (All) -- use format suggestions
fn_dom = "<PATH_TO_NEMO_DOMAIN_FILE>"
fn_dat = "<PATH_TO_NEMO_DATA_FILE(S)>"  # .format(run_name, start_date.year)
dn_out = "<PATH_TO_OUTPUT_DIRECTORY>"  # .format(run_name)
fn_prof = "<PATH_TO_PROCESSED_EN4_DATA>"
fn_cfg_nemo = "<PATH_TO_COAST_GRIDDED_CONFIG_FILE>"
fn_cfg_prof = "<PATH_TO_CODE_PROFILE_CONFIG_FILE>"

# CREATE NEMO OBJECT and read in NEMO data. Extract latitude and longitude array
print("Reading model data..", flush=True)
nemo = coast.Gridded(fn_dat, fn_dom, multiple=True, config=fn_cfg_nemo)
lon = nemo.dataset.longitude.values.squeeze()
lat = nemo.dataset.latitude.values.squeeze()
print("NEMO object created", flush=True)

# Extract time indices between start and end dates
nemo = nemo.time_slice(start_date, end_date)

# Create a landmask array -- important for obs_operator. We can
# get a landmask from bottom_level.
nemo.dataset["landmask"] = nemo.dataset.bottom_level == 0
nemo.dataset = nemo.dataset.rename({"depth_0": "depth"})
print("Landmask calculated", flush=True)

# CREATE EN4 PROFILE OBJECT
# If you have not already processed the data:
profile = coast.Profile(config=fn_cfg_prof)
profile.read_en4(fn_prof)
profile = profile.process_en4()

# If you have already processed then uncomment:
# profile = coast.Profile(dataset = xr.open_dataset(fn_prof, chunks={"id_dim": 10000}))
print("Profile object created", flush=True)

# Slice out the Profile times
profile = profile.time_slice(start_date, end_date)

# Extract only the variables that we want
nemo.dataset = nemo.dataset[["temperature", "bathymetry", "bottom_level", "landmask"]]
profile.dataset = profile.dataset[["potential_temperature", "practical_salinity", "depth"]]
profile.dataset = profile.dataset.rename({"potential_temperature": "temperature", "practical_salinity": "salinity"})

# Create Profile analysis object
profile_analysis = coast.ProfileAnalysis()

# Interpolate model to obs using obs_operator()
model_profiles = profile.obs_operator(nemo)
print("Obs_operator successful.", flush=True)

# Throw away profiles where the interpolation distance is larger than 5km.
keep_indices = model_profiles.dataset.interp_dist <= 5
model_profiles = model_profiles.isel(id_dim=keep_indices)
profile = profile.isel(id_dim=keep_indices)

# Load the profiles (careful with memory)
profile.dataset.load()
print("Model interpolated to obs locations", flush=True)

# Vertical Interpolation of model profiles to obs depths
model_profiles_interp = profile_analysis.interpolate_vertical(model_profiles, profile, interp_method="linear")
print("Model interpolated to obs depths", flush=True)

# Vertical interpolation of model profiles to reference depths
model_profiles_interp = profile_analysis.interpolate_vertical(model_profiles_interp, ref_depth)
model_profiles.dataset.to_netcdf(
    dn_out
    + "extracted_profiles_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)
model_profiles_interp.dataset.to_netcdf(
    dn_out
    + "interpolated_profiles_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)
print("Model interpolated to ref depths", flush=True)

# Interpolation of obs profiles to reference depths
profile_interp = profile_analysis.interpolate_vertical(profile, ref_depth)
profile_interp.dataset.to_netcdf(
    dn_out + "interpolated_obs_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)
print("Obs interpolated to reference depths", flush=True)

# Difference between Model and Obs
differences = profile_analysis.difference(profile_interp, model_profiles_interp)
differences.dataset.load()
differences.dataset.to_netcdf(
    dn_out + "profile_errors_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)
print("Calculated errors and written", flush=True)

# Surface Values and errors
model_profiles_surface = profile_analysis.depth_means(model_profiles, [0, surface_def])
obs_profiles_surface = profile_analysis.depth_means(profile, [0, surface_def])
surface_errors = profile_analysis.difference(obs_profiles_surface, model_profiles_surface)
surface_data = xr.merge(
    (surface_errors.dataset, model_profiles_surface.dataset, obs_profiles_surface.dataset), compat="override"
)
surface_data.to_netcdf(
    dn_out + "surface_data_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)

# Bottom values and errors
model_profiles_bottom = profile_analysis.bottom_means(model_profiles, bottom_height, bottom_thresh)
obs_bathymetry = model_profiles.dataset["bathymetry"].values
profile.dataset["bathymetry"] = (["id_dim"], obs_bathymetry)
obs_profiles_bottom = profile_analysis.bottom_means(profile, bottom_height, bottom_thresh)
bottom_errors = profile_analysis.difference(model_profiles_bottom, obs_profiles_bottom)
bottom_data = xr.merge(
    (bottom_errors.dataset, model_profiles_bottom.dataset, obs_profiles_bottom.dataset), compat="override"
)
bottom_data.to_netcdf(
    dn_out + "bottom_data_{0}_{1}_{2}.nc".format(run_name, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
)
print("Bottom and surface data estimated and written", flush=True)
print("DONE", flush=True)