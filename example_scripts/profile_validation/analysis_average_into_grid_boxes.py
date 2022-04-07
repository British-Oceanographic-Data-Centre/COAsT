"""
Script for showing use of Profile.average_into_grid_boxes(). This routines
takes all data in a Profile obejct and averages it into lat/lon grid boxes.

This script can be used for comparing observed and modelled climatologies. 
It should be run AFTER the nearest profiles have been extracted from the model
data, such as shown in analysis_extract_and_compare.py. 

Input and output files should be provided as a list. If you only have
one input file, then just enclose the string in []. 

You can use this script with example files by setting:
    
    # fn_prof = path.join('./example_files', "coast_example_EN4_201008.nc")
    # fn_cfg_prof = path.join('./config', "example_en4_profiles.json")
    # fn_out = path.join('./example_files', 'mask_mean.nc')
"""

import coast
import numpy as np
import xarray as xr
import os

# List of input files
fn_prof = "<PATH_TO_NETCDF_PROFILE_OBJECT>"
fn_cfg_prof = "<PATH_TO_COAST_PROFILE_CONFIG_FILE_IF_NEEDED>"

# Names of output files (coresponding to fn_in_list), include ".nc"
fn_out = "<OUTPUT_FILENAME.nc>"

# Define longitude and latitude grid
grid_lon = np.arange(-15, 15, 0.5)
grid_lat = np.arange(45, 65, 0.5)

# Load in data for averaging (e.g. surface data). 
prof_data = coast.Profile(config = fn_cfg_prof)
prof_data.read_en4(fn_prof)
profile_analysis = coast.ProfileAnalysis()

# Take just the data we want so it is faster
prof_data.dataset = prof_data.dataset[['temperature','practical_salinity']]

# Average all data across all seasons
prof_gridded = profile_analysis.average_into_grid_boxes(prof_data, grid_lon, grid_lat)

# Average data for each season
prof_gridded_DJF = profile_analysis.average_into_grid_boxes(prof_data, grid_lon, grid_lat, season="DJF", var_modifier="_DJF")
prof_gridded_MAM = profile_analysis.average_into_grid_boxes(prof_data, grid_lon, grid_lat, season="MAM", var_modifier="_MAM")
prof_gridded_JJA = profile_analysis.average_into_grid_boxes(prof_data, grid_lon, grid_lat, season="JJA", var_modifier="_JJA")
prof_gridded_SON = profile_analysis.average_into_grid_boxes(prof_data, grid_lon, grid_lat, season="SON", var_modifier="_SON")

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
