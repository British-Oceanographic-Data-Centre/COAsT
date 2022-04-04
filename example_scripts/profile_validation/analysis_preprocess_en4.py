#!/usr/bin/env python3
# Script for processing raw EN4 data prior to analysis.
# See docstring of Profile.process_en4() for more specifics on what it does.
#
# This script will just load modules, read in raw EN4 data, cut out
# a geographical box, call the processing routine and write the
# processed data to file.
#
# You don't have to do this for each EN4 file individually if you
# don't want, you can read in multiple using multiple = True on
# the creation of the profile object. However, if analysing model
# data in parallel chunks, you may want to split up the processing
# into smaller files to make the analysis faster.

### Start script
import sys

# IF USING A DEVELOPMENT BRANCH OF COAST, ADD THE REPOSITORY TO PATH:
# sys.path.append('<PATH_TO_COAST_REPO')
import coast
import pandas as pd

print("Modules loaded")

# File paths - input en4, output processed file and read config file
fn_en4 = "<PATH_TO_RAW_EN4_DATA_FILE(S)>"
fn_out = "<PATH_TO_OUTPUT_LOCATION_FOR_PROCESSED_PROFILES>"
fn_cfg_prof = "<PATH_TO_COAST_PROFILE_CONFIG_FILE>"

# Some important settings, easier to get at here
longitude_bounds = [-15, 15]  # Geo box to cut out from data (match to model)
latitude_bounds = [40, 65]
multiple = True  # Reading multple files?


# Create profile object containing data
profile = coast.Profile(config=fn_cfg_prof)
profile.read_en4(fn_en4, multiple=multiple)

# Get geographical indices to extract
profile = profile.subset_indices_lonlat_box(longitude_bounds, latitude_bounds)

# Process the extracted data into new processed profile
processed_profile = profile.process_en4()

# Not sure why but this is needed for now (xarray issue?)
processed_profile.dataset["time"] = ("id_dim", pd.to_datetime(processed_profile.dataset.time.values))

# Write processed profiles to file
processed_profile.dataset.to_netcdf(fn_out)
