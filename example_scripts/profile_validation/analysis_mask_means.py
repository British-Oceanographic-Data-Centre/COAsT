"""
Calculate mask means (regional means) of variables within a Profile object.

Provide paths to four files:
    
    fn_dom : NEMO domain file defining mask lon/lat.
    fn_cfg_nemo : NEMO config file.
    fn_profile : Path to netCDF containing profile data.
    fn_out : Path to netCDF output file.
    
* This script is set up for mask regions within the AMM model region.

You can use this script with example files by setting:
    
    # fn_dom = path.join('./example_files', "coast_example_nemo_domain.nc")
    # fn_prof = path.join('./example_files', "coast_example_EN4_201008.nc")
    # fn_cfg_nemo = path.join('./config', "example_nemo_grid_t.json")
    # fn_cfg_prof = path.join('./config', "example_en4_profiles.json")
"""

import coast
import numpy as np

fn_dom = "<PATH_TO_NEMO_DOMAIN_FILE>"
fn_cfg_nemo = "<PATH_TO_COAST_GRIDDED_CONFIG_FILE>"
fn_cfg_prof = "<PATH_TO_COAST_PROFILE_CONFIG_FILE>"
fn_prof = "<PATH_TO_COAST_PROFILE_NETCDF>"
fn_out = "<PATH_TO_OUTPUT_FILE>"

# CREATE NEMO OBJECT and read in NEMO data. Extract latitude and longitude array
print("Reading model data..", flush=True)
nemo = coast.Gridded(fn_domain=fn_dom, multiple=True, config=fn_cfg_nemo)
lon = nemo.dataset.longitude.values.squeeze()
lat = nemo.dataset.latitude.values.squeeze()
print("NEMO object created", flush=True)

# Create analysis object and mask maker object
profile_analysis = coast.ProfileAnalysis()

# Make Profile object from Profile data or analysis data from analysis_extract_and_compare
# Here let's make it from EN4 data:
profile = coast.Profile(config=fn_cfg_prof)
profile.read_en4(fn_prof)

# Or if reading analysis straight from netCDF, uncomment the following
# profile = coast.Profile(dataset=xr.open_mfdataset(fn_prof, chunks={"id_dim": 10000}))

# Make MaskMaker object
mm = coast.MaskMaker()

print("Doing regional analysis..")
# Define Regional Masks
regional_masks = []
bath = nemo.dataset.bathymetry.values
regional_masks.append(np.ones(lon.shape))
regional_masks.append(mm.region_def_nws_north_sea(lon, lat, bath))
regional_masks.append(mm.region_def_nws_outer_shelf(lon, lat, bath))
regional_masks.append(mm.region_def_nws_english_channel(lon, lat, bath))
regional_masks.append(mm.region_def_nws_norwegian_trench(lon, lat, bath))
regional_masks.append(mm.region_def_kattegat(lon, lat, bath))
regional_masks.append(mm.region_def_south_north_sea(lon, lat, bath))
off_shelf = mm.region_def_off_shelf(lon, lat, bath)
off_shelf[regional_masks[3].astype(bool)] = 0
off_shelf[regional_masks[4].astype(bool)] = 0
regional_masks.append(off_shelf)
regional_masks.append(mm.region_def_irish_sea(lon, lat, bath))

region_names = [
    "whole_domain",
    "north_sea",
    "outer_shelf",
    "eng_channel",
    "nor_trench",
    "kattegat",
    "southern_north_sea",
    "irish_sea",
    "off_shelf",
]

mask_list = mm.make_mask_dataset(lon, lat, regional_masks)
mask_indices = profile_analysis.determine_mask_indices(profile, mask_list)

# Do mask averaging
mask_means = profile_analysis.mask_means(profile, mask_indices)
print("Regional means calculated.")

# SAVE mask dataset to file
mask_means.to_netcdf(fn_out)
print("done")
