"""
Files definitions to use for unit testing.

You can easily use any of these files in a test class by importing this file.
For example:
    
    import unit_test_files
    
    get_dom = unit_test_files.fn_nemo_dom
"""

import os.path as path

# All directories relative to unit_testing diretory
dn_files = "./example_files/"  # Example data directory
dn_config = "./config"  # Example config file directory
dn_fig = "./unit_testing/figures/"  # Figure saving directory
dn_scripts = "./example_scripts"  # Example scripts directory

# Check dn_files directory
if not path.isdir(dn_files):
    print("please go download the examples file from https://linkedsystems.uk/erddap/files/COAsT_example_files/")
    dn_files = input("what is the path to the example files:\n")
    if not path.isdir(dn_files):
        print(f"location f{dn_files} cannot be found")

# Data files
fn_nemo_grid_t_dat_summer = path.join(dn_files, "nemo_data_T_grid_Aug2015.nc")
fn_nemo_grid_t_dat = path.join(dn_files, "nemo_data_T_grid.nc")
fn_nemo_grid_u_dat = path.join(dn_files, "nemo_data_U_grid.nc")
fn_nemo_grid_v_dat = path.join(dn_files, "nemo_data_V_grid.nc")
fn_nemo_dat = path.join(dn_files, "coast_example_nemo_data.nc")
fn_nemo_dat_subset = path.join(dn_files, "coast_example_nemo_subset_data.nc")
file_names_amm7 = path.join(dn_files, "nemo_data_T_grid*.nc")
fn_altimetry = path.join(dn_files, "coast_example_altimetry_data.nc")
fn_tidegauge = path.join(dn_files, "tide_gauges/lowestoft-p024-uk-bodc")
fn_tidegauge2 = path.join(dn_files, "tide_gauges/LIV2010.txt")
fn_multiple_tidegauge = path.join(dn_files, "tide_gauges/l*")
fn_gladstone = path.join(dn_files, "Gladstone_2020-10_HLW.txt")
fn_nemo_harmonics = path.join(dn_files, "coast_nemo_harmonics.nc")
fn_nemo_harmonics_dom = path.join(dn_files, "coast_nemo_harmonics_dom.nc")
fn_profile = path.join(dn_files, "coast_example_EN4_201008.nc")

# Domain files
fn_nemo_dom = path.join(dn_files, "coast_example_nemo_domain.nc")

# Configuration files
fn_profile_config = path.join(dn_config, "example_en4_profiles.json")
fn_altimetry_config = path.join(dn_config, "example_altimetry.json")
fn_config_t_grid = path.join(dn_config, "example_nemo_grid_t.json")
fn_config_f_grid = path.join(dn_config, "example_nemo_grid_f.json")
fn_config_u_grid = path.join(dn_config, "example_nemo_grid_u.json")
fn_config_v_grid = path.join(dn_config, "example_nemo_grid_v.json")
fn_config_w_grid = path.join(dn_config, "example_nemo_grid_w.json")
