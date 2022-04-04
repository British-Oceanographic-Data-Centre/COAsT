"""
Script for showing use of Profile.average_into_grid_boxes(). This routines
takes all data in a Profile obejct and averages it into lat/lon grid boxes.

This script can be used for comparing observed and modelled climatologies. 
It should be run AFTER the nearest profiles have been extracted from the model
data, such as shown in analysis_extract_and_compare.py. 

Input and output files should be provided as a list. If you only have
one input file, then just enclose the string in []. 
"""

import coast
import numpy as np
import xarray as xr
import os

average_extracted_model_data = True

# List of input files
fn_in_list = ["<PATH_TO_NETCDF_PROFILE_OBJECT_1>", "<PATH_TO_NETCDF_PROFILE_OBJECT_2>", "..."]

# Directory to save output
dn_out = "<PATH_TO_OUTPUT_DIRECTORY>"

# Names of output files (coresponding to fn_in_list), include ".nc"
fn_out_list = ["<OUTPUT_FILENAME_1.nc>", "<OUTPUT_FILENAME_2.nc>", "..."]

# Define longitude and latitude grid
grid_lon = np.arange(-15, 15, 0.5)
grid_lat = np.arange(45, 65, 0.5)

#%%

number_of_files = len(fn_in_list)

for ff in range(number_of_files):

    fn_in = fn_in_list[ff]
    fn_out = os.path.join(dn_out, fn_out_list[ff])

    # Load in data for averaging (e.g. surface data)
    prof_data = coast.Profile(fn_in)
    profile_analysis = coast.ProfileAnalysis()

    # Average all data across all seasons
    prof_gridded = prof_data.average_into_grid_boxes(grid_lon, grid_lat)

    # Average data for each season
    prof_gridded_DJF = profile_analysis.average_into_grid_boxes(grid_lon, grid_lat, season="DJF", var_modifier="_DJF")
    prof_gridded_MAM = profile_analysis.average_into_grid_boxes(grid_lon, grid_lat, season="MAM", var_modifier="_MAM")
    prof_gridded_JJA = profile_analysis.average_into_grid_boxes(grid_lon, grid_lat, season="JJA", var_modifier="_JJA")
    prof_gridded_SON = profile_analysis.average_into_grid_boxes(grid_lon, grid_lat, season="SON", var_modifier="_SON")

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
