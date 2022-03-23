##
"""
Script to do unit testing
Written as procedural code that plods through the code snippets and tests the
outputs or expected metadata.
***SECTIONS***
This script is separated into Sections and Subsections, for which there are two
counters to keep track: sec and subsec respectively.  At the beginning of each
section, the sec counter should be incremented by 1 and the subsec counter
should be reset to 96 (code for one below 'a'). At the beginning of each
subsection, subsec should be incremented by one.
***OTHER FILES***
There are two accompaniment files to this unit testing script:
    - unit_test_contents: A list of sections and subsections.
    - unit_test_guidelines: Further guidelines to creating unit tests.
Run:
ipython: cd COAsT; run unit_testing/unit_test.py  # I.e. from the git repo.
Unit template:
#-----------------------------------------------------------------------------#
# ( ## ) Subsection title                                                     #
#                                                                             #
subsec = subsec+1
# <Introduction>
try:
    # Do a thing
    #TEST: <description here>
    check1 = #<Boolean>
    check2 = #<Boolean>
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - ")
    else:
        print(str(sec) + chr(subsec) + " X - ")
except:
    print(str(sec) + chr(subsec) +' FAILED.')
"""

import coast
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import os.path as path
import logging
import coast.general_utils as general_utils
import coast.plot_util as plot_util
import coast.stats_util as stats_util
from socket import gethostname  # to get hostname
import traceback
import xarray.ufuncs as uf

"""
#################################################
## ( 0 ) Files, directories for unit testing   ##
#################################################
"""
## Initialise logging and save to log file
log_file = open("unit_testing/unit_test.log", "w")  # Need log_file.close()
coast.logging_util.setup_logging(stream=log_file, level=logging.CRITICAL)
## Alternative logging levels
# ..., level=logging.DEBUG) # Detailed information, typically of interest only when diagnosing problems.
# ..., level=logging.INFO) # Confirmation that things are working as expected.
# ..., level=logging.WARNING) # An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
# ..., level=logging.ERROR) # Due to a more serious problem, the software has not been able to perform some function
# ..., level=logging.CRITICAL) # A serious error, indicating that the program itself may be unable to continue running

dn_files = "./example_files/"

if not os.path.isdir(dn_files):
    print("please go download the examples file from https://linkedsystems.uk/erddap/files/COAsT_example_files/")
    dn_files = input("what is the path to the example files:\n")
    if not os.path.isdir(dn_files):
        print(f"location f{dn_files} cannot be found")

dn_fig = "unit_testing/figures/"
fn_nemo_grid_t_dat_summer = "nemo_data_T_grid_Aug2015.nc"
fn_nemo_grid_t_dat = "nemo_data_T_grid.nc"
fn_nemo_grid_u_dat = "nemo_data_U_grid.nc"
fn_nemo_grid_v_dat = "nemo_data_V_grid.nc"
fn_nemo_dat = "coast_example_nemo_data.nc"
fn_nemo_dat_subset = "coast_example_nemo_subset_data.nc"
fn_nemo_dom = "coast_example_nemo_domain.nc"
fn_altimetry = "coast_example_altimetry_data.nc"
fn_tidegauge = dn_files + "tide_gauges/lowestoft-p024-uk-bodc"
fn_tidegauge2 = dn_files + "tide_gauges/LIV2010.txt"
fn_nemo_harmonics = "coast_nemo_harmonics.nc"
fn_nemo_harmonics_dom = "coast_nemo_harmonics_dom.nc"
# EN4 profile data (NetCDF)
fn_profile = dn_files + "coast_example_EN4_201008.nc"
fn_profile_config = "config/example_en4_profiles.json"
fn_config_t_grid = path.join("./config", "example_nemo_grid_t.json")
fn_config_f_grid = path.join("./config", "example_nemo_grid_f.json")
fn_config_u_grid = path.join("./config", "example_nemo_grid_u.json")
fn_config_v_grid = path.join("./config", "example_nemo_grid_v.json")
fn_config_w_grid = path.join("./config", "example_nemo_grid_w.json")

sec = 1
subsec = 96  # Code for '`' (1 below 'a')
"""
#################################################
## ( 1 ) Loading/Initialisation                ##
#################################################
"""
# This section is for testing the loading and initialisation of Gridded objects.

# -----------------------------------------------------------------------------#
# %% ( 1a ) Load example Gridded Nemo data (Temperature, Salinity, SSH)                  #
#                                                                             #

subsec = subsec + 1

try:
    sci = coast.Gridded(path.join(dn_files, fn_nemo_dat), path.join(dn_files, fn_nemo_dom), config=fn_config_t_grid)

    # Test the data has loaded
    sci_attrs_ref = dict(
        [
            ("name", "AMM7_1d_20070101_20070131_25hourm_grid_T"),
            ("description", "ocean T grid variables, 25h meaned"),
            ("title", "ocean T grid variables, 25h meaned"),
            ("Conventions", "CF-1.6"),
            ("timeStamp", "2019-Dec-26 04:35:28 GMT"),
            ("uuid", "96cae459-d3a1-4f4f-b82b-9259179f95f7"),
        ]
    )

    # checking is LHS is a subset of RHS
    if sci_attrs_ref.items() <= sci.dataset.attrs.items():
        print(str(sec) + chr(subsec) + " OK - Nemo data loaded: " + fn_nemo_dat)
    else:
        print(str(sec) + chr(subsec) + " X - There is an issue with loading " + fn_nemo_dat)
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 1b ) Load data from existing dataset                                      #
#                                                                             #

subsec = subsec + 1
try:
    ds = xr.open_dataset(dn_files + fn_nemo_dat)
    sci_load_ds = coast.Gridded(config=fn_config_t_grid)
    sci_load_ds.load_dataset(ds)
    sci_load_file = coast.Gridded(config=fn_config_t_grid)
    sci_load_file.load(dn_files + fn_nemo_dat)
    if sci_load_ds.dataset.identical(sci_load_file.dataset):
        print(str(sec) + chr(subsec) + " OK - coast.load_dataset()")
    else:
        print(
            str(sec)
            + chr(subsec)
            + " X - coast.load_dataset() ERROR - not identical to dataset loaded via coast.load()"
        )
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 1c ) Set Gridded variable name                                               #
#                                                                             #

subsec = subsec + 1
try:
    sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)
    try:
        sci.dataset.temperature
    except NameError:
        print(str(sec) + chr(subsec) + " X - variable name (to temperature) not reset")
    else:
        print(str(sec) + chr(subsec) + " OK - variable name reset (to temperature)")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 1d ) Set Gridded grid attributes - dimension names                           #
#                                                                             #

subsec = subsec + 1
try:
    if sci.dataset.temperature.dims == ("t_dim", "z_dim", "y_dim", "x_dim"):
        print(str(sec) + chr(subsec) + " OK - dimension names reset")
    else:
        print(str(sec) + chr(subsec) + " X - dimension names not reset")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 1e ) Load only domain data in Gridded                                        #
#                                                                             #

subsec = subsec + 1

pass_test = False
nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_f_grid)

if nemo_f.dataset._coord_names == {"depth_0", "latitude", "longitude"}:
    var_name_list = []
    for var_name in nemo_f.dataset.data_vars:
        var_name_list.append(var_name)
    if var_name_list == ["bathymetry", "e1", "e2", "e3_0"]:
        pass_test = True

if pass_test:
    print(str(sec) + chr(subsec) + " OK - Gridded loaded domain data only")
else:
    print(str(sec) + chr(subsec) + " X - Gridded didn't load domain data correctly")

# -----------------------------------------------------------------------------#
# %% ( 1f ) Calculate depth_0 for t,u,v,w,f grids                                #
#                                                                             #

subsec = subsec + 1

try:
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )
    if not np.isclose(np.nansum(nemo_t.dataset.depth_0.values), 1705804300.0):
        raise ValueError(" X - Gridded depth_0 failed on t-grid failed")
    nemo_u = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_u_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_u_grid
    )
    if not np.isclose(np.nansum(nemo_u.dataset.depth_0.values), 1705317600.0):
        raise ValueError(" X - Gridded depth_0 failed on u-grid failed")
    nemo_v = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_v_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_v_grid
    )
    if not np.isclose(np.nansum(nemo_v.dataset.depth_0.values), 1705419100.0):
        raise ValueError(" X - Gridded depth_0 failed on v-grid failed")
    nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_f_grid)
    if not np.isclose(np.nansum(nemo_f.dataset.depth_0.values), 1704932600.0):
        raise ValueError(" X - Gridded depth_0 failed on f-grid failed")

    print(str(sec) + chr(subsec) + " OK - Gridded depth_0 calculations correct")
except ValueError as err:
    print(str(sec) + chr(subsec) + str(err))

# -----------------------------------------------------------------------------#
# %% ( 1g ) Load a subregion dataset with a full domain (AMM7)                   #
#                                                                             #

subsec = subsec + 1

try:

    amm7 = coast.Gridded(dn_files + fn_nemo_dat_subset, dn_files + fn_nemo_dom, config=fn_config_t_grid)

    # checking all the coordinates mapped correctly to the dataset object
    if amm7.dataset._coord_names == {"depth_0", "latitude", "longitude", "time"}:
        print(
            str(sec) + chr(subsec) + " OK - Gridded data subset loaded ", "with correct coords: " + fn_nemo_dat_subset
        )
    else:
        print(
            str(sec) + chr(subsec) + " X - There is an issue with ",
            "loading and subsetting the data " + fn_nemo_dat_subset,
        )

except:
    print(str(sec) + chr(subsec) + " FAILED. Test data in: {}.".format(fn_nemo_dat_subset))

# -----------------------------------------------------------------------------#
# %% ( 1h ) Load and combine (by time) multiple files  (AMM7)                    #
#                                                                             #

subsec = subsec + 1

try:
    file_names_amm7 = "nemo_data_T_grid*.nc"
    amm7 = coast.Gridded(dn_files + file_names_amm7, dn_files + fn_nemo_dom, config=fn_config_t_grid, multiple=True)

    # checking all the coordinates mapped correctly to the dataset object
    if amm7.dataset.time.size == 14:
        print(str(sec) + chr(subsec) + " OK - Gridded data loaded combine ", "over time: " + file_names_amm7)
    else:
        print(str(sec) + chr(subsec) + " X - There is an issue with loading", "multiple data files " + file_names_amm7)

except:
    print(str(sec) + chr(subsec) + " FAILED. Test data in: {} on {}.".format(dn_files, file_names_amm7))

# -----------------------------------------------------------------------------#
# %% ( 1i ) Load and combine harmonics                                         #
#                                                                             #

subsec = subsec + 1
# Load in a Gridded data file containing harmonics and combine them into a new
# Gridded object and dataset.

try:
    harmonics = coast.Gridded(dn_files + fn_nemo_harmonics, dn_files + fn_nemo_harmonics_dom, config=fn_config_t_grid)
    constituents = ["K1", "M2", "S2", "K2"]
    harmonics_combined = harmonics.harmonics_combine(constituents)

    # TEST: Check values in arrays and constituents
    check1 = list(harmonics_combined.dataset.constituent.values) == constituents
    check2 = harmonics_combined.dataset.harmonic_x[1].values == harmonics.dataset.M2x.values
    if check1 and check2.all():
        print(str(sec) + chr(subsec) + " OK - Harmonics loaded and combined")
    else:
        print(str(sec) + chr(subsec) + " X - Problem combining harmonics")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 1j ) Convert harmonics to a/g and back                                  #
#                                                                             #

subsec = subsec + 1
# Convert the harmonics loaded in 1i to amplitude and phase

try:
    harmonics_combined.harmonics_convert(direction="cart2polar")
    harmonics_combined.harmonics_convert(direction="polar2cart", x_var="x_test", y_var="y_test")

    # TEST: Check variables and differences
    check1 = "x_test" in harmonics_combined.dataset.keys()
    diff = harmonics_combined.dataset.harmonic_x[0].values - harmonics_combined.dataset.x_test[0].values
    check2 = np.max(np.abs(diff)) < 1e-6
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Harmonics converted")
    else:
        print(str(sec) + chr(subsec) + " X - Problem converting harmonics")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 1k ) Compute e3 from SSH field                                      #
#
subsec = subsec + 1
try:
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )

    e3t, e3u, e3v, e3f, e3w = coast.Gridded.get_e3_from_ssh(nemo_t, True, True, True, True, True)
    cksum = np.array([e3t.sum(), e3u.sum(), e3v.sum(), e3f.sum(), e3w.sum()])
    # these references are based on the example file's ssh field
    reference = np.array([8.337016e08, 8.333972e08, 8.344886e08, 8.330722e08, 8.265948e08])
    if np.allclose(cksum, reference):
        print(str(sec) + chr(subsec) + " OK - computed e3[t,u,v,f,w] as expected")
    else:
        print(str(sec) + chr(subsec) + " X - computed e3[t,u,v,f,w] not as expected")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

"""
#################################################
## ( 2 ) Test general utility methods in Coast ##
#################################################
"""
sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# %% ( 2a ) Copying a Coast object                                               #
#                                                                             #

subsec = subsec + 1

try:
    sci_copy = sci.copy()
    if sci_copy.dataset == sci.dataset:
        print(str(sec) + chr(subsec) + " OK - Copied Coast object ")
    else:
        print(str(sec) + chr(subsec) + " X - Copy Failed ")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 2b ) Coast __getitem__ returns variable                                   #
#                                                                             #

subsec = subsec + 1

try:
    if sci.dataset["ssh"].equals(sci["ssh"]):
        print(str(sec) + chr(subsec) + " OK - coast.__getitem__ works correctly ")
    else:
        print(str(sec) + chr(subsec) + " X - Problem with coast.__getitem__ ")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 2c ) Renaming variables inside a Coast object                             #
#                                                                             #

subsec = subsec + 1
try:
    sci_copy.rename({"ssh": "renamed"})
    if sci["ssh"].equals(sci_copy["renamed"]):
        print(str(sec) + chr(subsec) + " OK - Renaming of variable in dataset ")
    else:
        print(str(sec) + chr(subsec) + " X - Variable renaming failed ")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 2d ) day of the week function                                           #
#                                                                             #

subsec = subsec + 1
try:
    check = general_utils.day_of_week(np.datetime64("2020-10-16")) == "Fri"

    if check:
        print(str(sec) + chr(subsec) + " OK - day of the week method")
    else:
        print(str(sec) + chr(subsec) + " X - day of the week method")

except:
    print(str(sec) + chr(subsec) + " FAILED: day of the week method")

"""
#################################################
## ( 3 ) Test Diagnostic methods               ##
#################################################
"""
sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# %% ( 3a ) Computing a vertical spatial derivative                              #
#                                                                             #

subsec = subsec + 1

# Initialise DataArrays
nemo_t = coast.Gridded(fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid)
nemo_w = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_w_grid)

try:
    log_str = ""
    # Compute dT/dz
    nemo_w_1 = nemo_t.differentiate("temperature", config_path=fn_config_w_grid, dim="z_dim")
    if nemo_w_1 is None:  # Test whether object was returned
        log_str += "No object returned\n"
    # Make sure the hardwired grid requirements are present
    if not hasattr(nemo_w.dataset, "depth_0"):
        log_str += "Missing depth_0 variable\n"
    if not hasattr(nemo_w.dataset, "e3_0"):
        log_str += "Missing e3_0 variable\n"
    if not hasattr(nemo_w.dataset.depth_0, "units"):
        log_str += "Missing depth units\n"
    # Test attributes of derivative. This are generated last so can indicate earlier problems
    nemo_w_2 = nemo_t.differentiate("temperature", dim="z_dim", out_var_str="dTdz", out_obj=nemo_w)
    if not nemo_w_2.dataset.dTdz.attrs == {"units": "degC/m", "standard_name": "dTdz"}:
        log_str += "Did not write correct attributes\n"
    # Test auto-naming derivative. Again test expected attributes.
    nemo_w_3 = nemo_t.differentiate("temperature", dim="z_dim", config_path=fn_config_w_grid)
    if not nemo_w_3.dataset.temperature_dz.attrs == {"units": "degC/m", "standard_name": "temperature_dz"}:
        log_str += "Problem with auto-naming derivative field\n"

    ## Test numerical calculation. Differentiate f(z)=-z --> -1
    # Construct a depth variable - needs to be 4D
    nemo_t.dataset["depth4D"], _ = xr.broadcast(nemo_t.dataset["depth_0"], nemo_t.dataset["temperature"])
    nemo_w_4 = nemo_t.differentiate("depth4D", dim="z_dim", out_var_str="dzdz", config_path=fn_config_w_grid)
    if not np.isclose(
        nemo_w_4.dataset.dzdz.isel(z_dim=slice(1, nemo_w_4.dataset.dzdz.sizes["z_dim"])).max(), -1
    ) or not np.isclose(nemo_w_4.dataset.dzdz.isel(z_dim=slice(1, nemo_w_4.dataset.dzdz.sizes["z_dim"])).min(), -1):
        log_str += "Problem with numerical derivative of f(z)=-z\n"

    if log_str == "":
        print(str(sec) + chr(subsec) + " OK - Gridded.differentiate (for d/dz) method passes all tests")
    else:
        print(str(sec) + chr(subsec) + " X - Gridded.differentiate method failed: " + log_str)

except:
    traceback.print_exc()
    print(str(sec) + chr(subsec) + " X - setting derivative attributes failed ")

# -----------------------------------------------------------------------------#
# %% ( 3b ) Construct density                                                    #
#                                                                             #

subsec = subsec + 1
nemo_t = coast.Gridded(fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid)
nemo_t.construct_density()
yt, xt, length_of_line = nemo_t.transect_indices([54, -15], [56, -12])

try:
    if not np.allclose(
        nemo_t.dataset.density.sel(x_dim=xr.DataArray(xt, dims=["r_dim"]), y_dim=xr.DataArray(yt, dims=["r_dim"]))
        .sum(dim=["t_dim", "r_dim", "z_dim"])
        .item(),
        11185010.518671108,
    ):
        raise ValueError(str(sec) + chr(subsec) + " X - Density incorrect")
    print(str(sec) + chr(subsec) + " OK - Density correct")
except ValueError as err:
    print(err)
densitycopy = nemo_t.dataset.density.sel(x_dim=xr.DataArray(xt, dims=["r_dim"]), y_dim=xr.DataArray(yt, dims=["r_dim"]))

# -----------------------------------------------------------------------------#
# %% ( 3c ) Construct pycnocline depth and thickness                             #
#                                                                             #

subsec = subsec + 1

nemo_t = None
nemo_w = None
nemo_t = coast.Gridded(dn_files + fn_nemo_grid_t_dat_summer, dn_files + fn_nemo_dom, config=fn_config_t_grid)
# create an empty w-grid object, to store stratification
nemo_w = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_w_grid)
try:
    log_str = ""
    # initialise Internal Tide object
    IT = coast.InternalTide(nemo_t, nemo_w)
    if IT is None:  # Test whether object was returned
        log_str += "No object returned\n"
    # Construct pycnocline variables: depth and thickness
    IT.construct_pycnocline_vars(nemo_t, nemo_w)

    if not hasattr(nemo_t.dataset, "density"):
        log_str += "Did not create density variable\n"
    if not hasattr(nemo_w.dataset, "rho_dz"):
        log_str += "Did not create rho_dz variable\n"

    if not hasattr(IT.dataset, "strat_1st_mom"):
        log_str += "Missing strat_1st_mom variable\n"
    if not hasattr(IT.dataset, "strat_1st_mom_masked"):
        log_str += "Missing strat_1st_mom_masked variable\n"
    if not hasattr(IT.dataset, "strat_2nd_mom"):
        log_str += "Missing strat_2nd_mom variable\n"
    if not hasattr(IT.dataset, "strat_2nd_mom_masked"):
        log_str += "Missing strat_2nd_mom_masked variable\n"
    if not hasattr(IT.dataset, "mask"):
        log_str += "Missing mask variable\n"

    # Check the calculations are as expected
    if (
        np.isclose(IT.dataset.strat_1st_mom.sum(), 3.74214231e08)
        and np.isclose(IT.dataset.strat_2nd_mom.sum(), 2.44203298e08)
        and np.isclose(IT.dataset.mask.sum(), 450580)
        and np.isclose(IT.dataset.strat_1st_mom_masked.sum(), 3.71876949e08)
        and np.isclose(IT.dataset.strat_2nd_mom_masked.sum(), 2.42926865e08)
    ):
        print(str(sec) + chr(subsec) + " OK - pyncocline depth and thickness good")

except:
    print(str(sec) + chr(subsec) + " X - computing pycnocline depth and thickness failed ")

# -----------------------------------------------------------------------------#
# %% ( 3d ) Plot pycnocline depth                                              #
#                                                                             #

subsec = subsec + 1
try:
    fig, ax = IT.quick_plot("strat_1st_mom_masked")
    fig.tight_layout()
    fig.savefig(dn_fig + "strat_1st_mom.png")
    print(str(sec) + chr(subsec) + " OK - pycnocline depth plot saved")
except:
    print(str(sec) + chr(subsec) + "X - quickplot() failed")

"""
#################################################
## ( 4 ) Test Transect related methods         ##
#################################################
"""
sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# %% ( 4a ) Determining and extracting transect indices                          #
#                                                                             #

subsec = subsec + 1

# Extract transect indices
nemo_t = coast.Gridded(fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid)
yt, xt, length_of_line = nemo_t.transect_indices([51, -5], [49, -9])

# Test transect indices
yt_ref = [
    164,
    163,
    162,
    162,
    161,
    160,
    159,
    158,
    157,
    156,
    156,
    155,
    154,
    153,
    152,
    152,
    151,
    150,
    149,
    148,
    147,
    146,
    146,
    145,
    144,
    143,
    142,
    142,
    141,
    140,
    139,
    138,
    137,
    136,
    136,
    135,
    134,
]
xt_ref = [
    134,
    133,
    132,
    131,
    130,
    129,
    128,
    127,
    126,
    125,
    124,
    123,
    122,
    121,
    120,
    119,
    118,
    117,
    116,
    115,
    114,
    113,
    112,
    111,
    110,
    109,
    108,
    107,
    106,
    105,
    104,
    103,
    102,
    101,
    100,
    99,
    98,
]
length_ref = 37


if (xt == xt_ref) and (yt == yt_ref) and (length_of_line == length_ref):
    print(str(sec) + chr(subsec) + " OK - Gridded transect indices extracted")
else:
    print(str(sec) + chr(subsec) + " X - Issue with transect indices extraction from Gridded")

# -----------------------------------------------------------------------------#
# %% ( 4b ) Transport velocity and depth calculations                            #
#
subsec = subsec + 1
try:
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )
    nemo_u = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_u_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_u_grid
    )
    nemo_v = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_v_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_v_grid
    )
    nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_f_grid)

    tran_f = coast.TransectF(nemo_f, (54, -15), (56, -12))
    tran_f.calc_flow_across_transect(nemo_u, nemo_v)
    cksum1 = tran_f.data_cross_tran_flow.normal_velocities.sum(dim=("t_dim", "z_dim", "r_dim")).item()
    cksum2 = tran_f.data_cross_tran_flow.normal_transports.sum(dim=("t_dim", "r_dim")).item()
    if np.isclose(cksum1, -253.6484375) and np.isclose(cksum2, -48.67562136873888):
        print(str(sec) + chr(subsec) + " OK - Transect cross flow calculations as expected")
    else:
        print(str(sec) + chr(subsec) + " X - Transect cross flow calculations not as expected")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

# -----------------------------------------------------------------------------#
# %% ( 4c ) Transport and velocity plotting                                      #
#
subsec = subsec + 1
try:
    fig, ax = tran_f.plot_transect_on_map()
    ax.set_xlim([-20, 0])  # Problem: nice to make the land appear.
    ax.set_ylim([45, 65])  # But can not call plt.show() before adjustments are made...
    # fig.tight_layout()
    fig.savefig(dn_fig + "transect_map.png")

    plot_dict = {"fig_size": (5, 3), "title": "Normal velocities"}
    fig, ax = tran_f.plot_normal_velocity(time=0, cmap="seismic", plot_info=plot_dict, smoothing_window=2)
    fig.tight_layout()
    fig.savefig(dn_fig + "transect_velocities.png")
    plot_dict = {"fig_size": (5, 3), "title": "Transport across AB"}
    fig, ax = tran_f.plot_depth_integrated_transport(time=0, plot_info=plot_dict, smoothing_window=2)
    fig.tight_layout()
    fig.savefig(dn_fig + "transect_transport.png")
    print(str(sec) + chr(subsec) + " OK - Transect velocity and transport plots saved")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

# -----------------------------------------------------------------------------#
# %% ( 4d ) Construct density and pressure along the transect                    #
#
subsec = subsec + 1
try:
    tran_t = coast.TransectT(nemo_t, (54, -15), (56, -12))
    tran_t.construct_pressure()
    cksum1 = tran_t.data.density_zlevels.sum(dim=["t_dim", "r_dim", "depth_z_levels"]).compute().item()
    cksum2 = tran_t.data.pressure_h_zlevels.sum(dim=["t_dim", "r_dim", "depth_z_levels"]).compute().item()
    cksum3 = tran_t.data.pressure_s.sum(dim=["t_dim", "r_dim"]).compute().item()
    if np.allclose([cksum1, cksum2, cksum3], [23800545.87457855, 135536478.93335825, -285918.5625]):
        print(str(sec) + chr(subsec) + " OK - Transect density and pressure calculations as expected")
    else:
        print(str(sec) + chr(subsec) + " X - Transect density and pressure calculations not as expected")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

# -----------------------------------------------------------------------------#
# %% ( 4e ) Calculate the geostrophic flow across the transect                   #
#
subsec = subsec + 1
try:
    tran_f.calc_geostrophic_flow(nemo_t, config_u=fn_config_u_grid, config_v=fn_config_v_grid)
    cksum1 = tran_f.data_cross_tran_flow.normal_velocity_hpg.sum(dim=("t_dim", "depth_z_levels", "r_dim")).item()
    cksum2 = tran_f.data_cross_tran_flow.normal_velocity_spg.sum(dim=("t_dim", "r_dim")).item()
    cksum3 = tran_f.data_cross_tran_flow.normal_transport_hpg.sum(dim=("t_dim", "r_dim")).item()
    cksum4 = tran_f.data_cross_tran_flow.normal_transport_spg.sum(dim=("t_dim", "r_dim")).item()

    if np.allclose([cksum1, cksum2, cksum3, cksum4], [84.8632969783, -5.09718418121, 115.2587369660, -106.7897376093]):
        print(str(sec) + chr(subsec) + " OK - Transect geostrophic flow calculations as expected")
    else:
        print(str(sec) + chr(subsec) + " X - Transect geostrophic flow calculations now as expected")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())
"""
#################################################
## ( 5 ) Object Manipulation (e.g. subsetting) ##
#################################################
"""
sec = sec + 1
subsec = 96
# -----------------------------------------------------------------------------#
# %% ( 5a ) Subsetting single variable                                           #
#                                                                             #
subsec = subsec + 1

try:
    # Extact the variable
    data_t = sci.get_subset_as_xarray("temperature", xt_ref, yt_ref)

    # Test shape and exteme values
    if (
        (np.shape(data_t) == (51, 37))
        and (np.nanmin(data_t) - 11.267578 < 1e-6)
        and (np.nanmax(data_t) - 11.834961 < 1e-6)
    ):
        print(
            str(sec)
            + chr(subsec)
            + " OK - Gridded Coast get_subset_as_xarray extracted expected array size and "
            + "extreme values"
        )
    else:
        print(str(sec) + chr(subsec) + " X - Issue with Gridded Coast get_subset_as_xarray method")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 5b ) Indices by distance method                                           #
#                                                                             #

subsec = subsec + 1

try:
    # Find indices for points with 111 km from 0E, 51N

    ind = sci.subset_indices_by_distance(0, 51, 111)

    # Test size of indices array
    if np.shape(ind) == (2, 674):
        print(
            str(sec)
            + chr(subsec)
            + " OK - Gridded domain subset_indices_by_distance extracted expected "
            + "size of indices"
        )
    else:

        print(
            str(sec)
            + chr(subsec)
            + "X - Issue with indices extraction from Gridded domain "
            + "subset_indices_by_distance method"
        )
except:
    print(str(sec) + chr(subsec) + " FAILED")


# -----------------------------------------------------------------------------#
# %% ( 5c ) Find nearest xy indices                                              #
#                                                                             #

subsec = subsec + 1
try:
    altimetry = coast.Altimetry(dn_files + fn_altimetry)
    ind = altimetry.subset_indices_lonlat_box([-10, 10], [45, 60])
    altimetry_nwes = altimetry.isel(time=ind)  # nwes = northwest europe shelf
    ind_x, ind_y = general_utils.nearest_indices_2d(
        sci.dataset.longitude, sci.dataset.latitude, altimetry_nwes.dataset.longitude, altimetry_nwes.dataset.latitude
    )
    if ind_x.shape == altimetry_nwes.dataset.longitude.shape:
        print(str(sec) + chr(subsec) + " OK - nearest_xy_indices works ")
    else:
        print(str(sec) + chr(subsec) + "X - Problem with nearest_xy_indices()")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 5d ) Interpolate in space (nearest)                                       #
#                                                                             #

subsec = subsec + 1
try:
    interp_lon = np.array(altimetry_nwes.dataset.longitude).flatten()
    interp_lat = np.array(altimetry_nwes.dataset.latitude).flatten()
    interpolated = sci.interpolate_in_space(sci.dataset.ssh, interp_lon, interp_lat)

    # Check that output array longitude has same shape as altimetry
    if interpolated.longitude.shape == altimetry_nwes.dataset.longitude.shape:
        print(str(sec) + chr(subsec) + " OK - Space interpolation works ")
    else:
        print(str(sec) + chr(subsec) + "X - Problem with space interpolation")
except:
    print(str(sec) + chr(subsec) + " FAILED")

# -----------------------------------------------------------------------------#
# %% ( 5e ) Interpolate in time                                                  #
#                                                                             #

subsec = subsec + 1
try:
    interpolated = sci.interpolate_in_time(interpolated, altimetry_nwes.dataset.time)

    # Check time in interpolated object has same shape
    if interpolated.time.shape == altimetry_nwes.dataset.time.shape:
        print(str(sec) + chr(subsec) + " OK - Time interpolation works ")
    else:
        print(str(sec) + chr(subsec) + "X - Problem with time interpolation")
except:
    print(str(sec) + chr(subsec) + " FAILED")


"""
#################################################
## ( 6 ) Altimetry Methods                     ##
#################################################
"""
sec = sec + 1
subsec = 96
# This section is for testing and demonstrating the use of the Altimetry
# object. First begin by reloading Gridded t-grid test data:
sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)


# -----------------------------------------------------------------------------#
# %% ( 6a ) Load example altimetry data                                          #
#                                                                             #

subsec = subsec + 1
# We can load altimetry data straight from a CMEMS netcdf file on initialisation
try:
    altimetry = coast.Altimetry(dn_files + fn_altimetry, config="./config/example_altimetry.json")

    # Test the data has loaded using attribute comparison, as for nemo_data
    alt_attrs_ref = dict(
        [
            ("source", "Jason-1 measurements"),
            ("date_created", "2019-02-20T11:20:56Z"),
            ("institution", "CLS, CNES"),
            ("Conventions", "CF-1.6"),
        ]
    )

    # checking is LHS is a subset of RHS
    if alt_attrs_ref.items() <= altimetry.dataset.attrs.items():
        print(str(sec) + chr(subsec) + " OK - Altimetry data loaded: " + fn_altimetry)
    else:
        print(str(sec) + chr(subsec) + " X - There is an issue with loading: " + fn_altimetry)
except:
    print(str(sec) + chr(subsec) + " FAILED")


# -----------------------------------------------------------------------------#
# %% ( 6b ) Altimetry subsetting                                                 #
#                                                                             #

subsec = subsec + 1
# The altimetry that we loaded is global so lets subset it for the North
# West European Shelf.
try:
    ind = altimetry.subset_indices_lonlat_box([-10, 10], [45, 60])
    ind = ind[::4]
    altimetry_nwes = altimetry.isel(t_dim=ind)  # nwes = northwest europe shelf

    if altimetry_nwes.dataset.dims["t_dim"] == 54:
        print(str(sec) + chr(subsec) + " OK - Altimetry object subsetted using isel ")
    else:
        print(str(sec) + chr(subsec) + "X - Failed to subset object/ return as copy")
except:
    print(str(sec) + chr(subsec) + " FAILED")


# -----------------------------------------------------------------------------#
# %% ( 6c ) Interpolate model to altimetry                                       #
#                                                                             #

subsec = subsec + 1
# Now lets interpolate a model variable to the altimetry space using
# obs_operator().

try:
    altimetry_nwes.obs_operator(sci, "ssh")
    # Check new variable is in altimetry dataset and isn't all NaNs
    try:
        test = altimetry_nwes.dataset.interp_ssh
        if False in np.isnan(altimetry_nwes.dataset.interp_ssh):
            print(str(sec) + chr(subsec) + " OK - model SSH interpolated to altimetry")
        else:
            print(str(sec) + chr(subsec) + " OK - X - Interpolation to altimetry failed")
    except:
        print(str(sec) + chr(subsec) + " X - Interpolation to altimetry failed")
except:
    print(str(sec) + chr(subsec) + " FAILED")


# -----------------------------------------------------------------------------#
# %% ( 6d ) Altimetry CRPS                                                       #
#                                                                             #


subsec = subsec + 1
# Compare modelled SSH to observed sea level using CRPS. This can be done
# either by creating a new object (create_new_object=True) or by saving it
# to the existing object. Here we look at the first option.

try:
    crps = altimetry_nwes.crps(sci, "ssh", "ocean_tide_standard_name")

    # TEST: Check length of crps and that it contains values
    check1 = crps.dataset.crps.shape[0] == altimetry_nwes.dataset.ocean_tide_standard_name.shape[0]
    check2 = False in np.isnan(crps.dataset.crps)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Altimetry CRPS")
    else:
        print(str(sec) + chr(subsec) + " X - Altimetry CRPS")

except:
    print(str(sec) + chr(subsec) + " FAILED.")


# -----------------------------------------------------------------------------#
# %% ( 6e ) Altimetry Stats methods                                              #
#                                                                             #

subsec = subsec + 1
# We can batch return the basic stats methods from Altimetry using basic_stats().
# We test all of the stats routines here by using this batch function.
# Here we compare an altimetry variable to our interpolate model SSH

try:
    stats = altimetry_nwes.basic_stats("ocean_tide_standard_name", "interp_ssh")
    altimetry_nwes.basic_stats("ocean_tide_standard_name", "interp_ssh", create_new_object=False)

    # TEST: Check new object resembles internal object
    check1 = all(stats.dataset.error == altimetry_nwes.dataset.error)
    # TEST: Check lengths and values
    check2 = stats.dataset.absolute_error.shape[0] == altimetry_nwes.dataset.ocean_tide_standard_name.shape[0]
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Basic Stats for Altimetry")
    else:
        print(str(sec) + chr(subsec) + " X -  Basic Stats for Altimetry")

except:
    print(str(sec) + chr(subsec) + " FAILED.")


# -----------------------------------------------------------------------------#
# %% ( 6f ) Altimetry quick_plot()                                               #
#                                                                             #

subsec = subsec + 1
# Now lets take a look at our CRPS values on a map
plt.close("all")

try:
    fig, ax = crps.quick_plot("crps")
    fig.savefig(dn_fig + "altimetry_crps_quick_plot.png")
    # plt.close(fig)
    print(str(sec) + chr(subsec) + " OK - Altimetry quick plot saved")
except:
    print(str(sec) + chr(subsec) + " X - Altimetry quick plot not saved")

plt.close("all")

"""
#################################################
## ( 7 ) Tidegauge Methods                     ##
#################################################
"""
sec = sec + 1
subsec = 96

# This section is for testing and demonstrating the use of the Tidegauge
# object. First begin by reloading Gridded t-grid test data:
sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)


# -----------------------------------------------------------------------------#
# %% ( 7a ) Load in GESLA tide gauge files from directory                      #
#                                                                             #

subsec = subsec + 1

# Tide gauge data can be loaded straight a GESLA data file.
# Here, I load in data for Lowestoft by providing a filename and restricting
# the dates to January 2007 using two datetime object:

try:
    date0 = datetime.datetime(2007, 1, 10)
    date1 = datetime.datetime(2007, 1, 12)
    lowestoft = coast.Tidegauge(fn_tidegauge, date_start=date0, date_end=date1)

    # TEST: Define Attribute dictionary for comparison
    test_attrs = {
        "site_name": "Lowestoft",
        "country": "United_Kingdom",
        "contributor": "British_Oceanographic_Data_Centre",
        "coordinate_system": "UK",
        "original_start_date": np.datetime64("1964-01-01 00:00:00"),
        "original_end_date": np.datetime64("2014-12-31 23:45:00"),
        "time_zone_hours": 0.0,
        "precision": 0.001,
        "null_value": -99.9999,
    }

    # TEST: Check attribute dictionary and length of sea_level.
    check1 = len(lowestoft.dataset.sea_level) == 193
    check2 = lowestoft.dataset.attrs == test_attrs
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Tide gauge loaded")
    else:
        print(str(sec) + chr(subsec) + " X - Failed to load tide gauge")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7b ) Load in BODC tide gauge data                                       #
#                                                                             #

subsec = subsec + 1


# Load and plot BODC processed data
try:
    # Set the start and end dates
    date_start = np.datetime64("2020-10-12 23:59")
    date_end = np.datetime64("2020-10-14 00:01")

    # Initiate a Tidegauge object, if a filename is passed it assumes it is a GESLA
    # type object
    tg = coast.Tidegauge()
    # specify the data read as a High Low Water dataset
    tg.dataset = tg.read_bodc_to_xarray(fn_tidegauge2, date_start, date_end)
    # tg.plot_timeseries()

    # TEST: Define Attribute dictionary for comparison
    test_attrs = {
        "port": "p234",
        "site": "liverpool,_gladstone_dock",
        "start_date": "01oct2020-00.00.00",
        "end_date": "31oct2020-23.45.00",
        "contributor": "national_oceanography_centre,_liverpool",
        "datum_information": "the_data_refer_to_admiralty_chart_datum_(acd)",
        "parameter_code": "aslvbg02_=_surface_elevation_(unspecified_datum)_of_the_water_body_by_bubbler_tide_gauge_(second_sensor)",
        "site_name": "liverpool,_gladstone_dock",
    }

    # TEST: Check attribute dictionary and length of sea_level.
    check1 = len(tg.dataset.sea_level) == 97
    check2 = tg.dataset.attrs == test_attrs
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - BODC tide gauge loaded")
    else:
        print(str(sec) + chr(subsec) + " X - Failed to load BODC tide gauge")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7c ) Load in Environment Agency river gauge data from API               #
#                                                                             #

subsec = subsec + 1

# Load in data obtained using the Environment Agency (England)
#  API. These are only accessible for the last 28 days. This does not require
# an API key.
#  Details of available tidal stations are recovered with:
#  https://environment.data.gov.uk/flood-monitoring/id/stations?type=TideGauge
# Recover the "stationReference" for the gauge of interest and pass as
# stationId:str. The default gauge is Liverpool: stationId="E70124"
# Construct a recent 10 days period and extract these data

try:
    date_start = np.datetime64("now") - np.timedelta64(20, "D")
    date_end = np.datetime64("now") - np.timedelta64(10, "D")
    eg = coast.Tidegauge()
    # Extract the data between explicit dates
    eg.dataset = eg.read_ea_api_to_xarray(date_start=date_start, date_end=date_end)
    check1 = eg.dataset.site_name == "Liverpool"
    check2 = len(eg.dataset.sea_level) > 0
    # eg.plot_timeseries()

    # Alternatively extract the data for the last n_days, here for a specific
    # (the default) station.
    eg.dataset = eg.read_ea_api_to_xarray(n_days=1, station_id="E70124")
    check3 = eg.dataset.site_name == "Liverpool"
    # eg.plot_timeseries()

    if check1 and check2 and check3:
        print(str(sec) + chr(subsec) + " OK - EA Tide gauge loaded")
    else:
        print(str(sec) + chr(subsec) + " X - Failed to load EA tide gauge")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7d ) Tidegauge obs_operator                                             #
#                                                                             #

subsec = subsec + 1
# Lets interpolate the sossheig variable in sci onto the location and times of
# our tidegauge using obs_operator(). Here, we interpolate linearly in time.

try:

    lowestoft.obs_operator(sci, "ssh", time_interp="linear", model_mask="bathy")

    # TEST: Check that the resulting interp_sossheig variable is of the same
    # length as sea_level and that it is populated.
    interp = lowestoft.dataset.interp_ssh
    interp_len = interp.shape[0]
    orig_len = lowestoft.dataset.sea_level.shape[0]
    check1 = interp_len == orig_len
    check2 = False in np.isnan(lowestoft.dataset.interp_ssh)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Tide gauge obs_operator")
    else:
        print(str(sec) + chr(subsec) + " X - Tide gauge obs_operator")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7e) Tidegauge CRPS                                                      #
#                                                                             #
subsec = subsec + 1
# Compare modelled SSH to observed sea level using CRPS. This can be done
# either by creating a new object (create_new_object=True) or by saving it
# to the existing object. Here we look at the first option.

try:
    crps = lowestoft.crps(sci, "ssh")

    # TEST: Check length of crps and that it contains values
    check1 = crps.dataset.crps.shape[0] == lowestoft.dataset.sea_level.shape[0]
    check2 = False in np.isnan(crps.dataset.crps)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - CRPS between gauge and model exists")
    else:
        print(str(sec) + chr(subsec) + " X - Problem with guage/model CRPS")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7f ) Tidegauge Stats methods                                            #
#                                                                             #
subsec = subsec + 1
# We can batch return the basic stats methods from Tidegauge using basic_stats().
# We test all of the stats routines here by using this batch function.

try:
    stats = lowestoft.basic_stats("sea_level", "interp_ssh")
    lowestoft.basic_stats("sea_level", "interp_ssh", create_new_object=False)

    # TEST: Check new object resembles internal object
    check1 = all(stats.dataset.error == lowestoft.dataset.error)
    # TEST: Check lengths and values
    check2 = stats.dataset.absolute_error.shape[0] == lowestoft.dataset.sea_level.shape[0]
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Basic Stats for Tidegauge")
    else:
        print(str(sec) + chr(subsec) + " X -  Basic Stats for Tidegauge")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7g ) Tidegauge Resample to hourly                                       #
#                                                                             #
subsec = subsec + 1
# Lets resample the tide gauge data to be hourly.

try:
    lowestoft.resample_mean("sea_level", "1H")

    # TEST: Check new times have right frequency
    td0 = lowestoft.dataset.time_1H[1] - lowestoft.dataset.time_1H[0]
    check1 = td0.values.astype("timedelta64[h]") == np.timedelta64(1, "h")
    # TEST: Check length
    check2 = np.ceil(lowestoft.dataset.time.shape[0] / 4) == lowestoft.dataset.time_1H.shape[0]
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Tidegauge resampled")
    else:
        print(str(sec) + chr(subsec) + " X -  Resample Tidegauge")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7h ) Apply Doodson XO filter to hourly data                             #
#                                                                             #
subsec = subsec + 1
# Lets resample the tide gauge data to be hourly.

try:
    lowestoft.apply_doodson_x0_filter("sea_level_1H")

    # TEST: Check new times are same length as variable
    check1 = lowestoft.dataset.time_1H.shape == lowestoft.dataset.sea_level_1H_dx0.shape
    # TEST: Check there are number values in output
    check2 = False in np.isnan(lowestoft.dataset.sea_level_1H_dx0)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Tidegauge doodson X0")
    else:
        print(str(sec) + chr(subsec) + " X -  Tidegauge doodson X0")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7i ) Tidegauge Loading multiple TidegaugeS                              #
#                                                                             #
subsec = subsec + 1
# We can load multiple tide gauges into a list of Tidegauge objects using the
# static method create_multiple_tidegauge.

try:
    date0 = datetime.datetime(2007, 1, 10)
    date1 = datetime.datetime(2007, 1, 12)
    tidegauge_list = coast.Tidegauge.create_multiple_tidegauge("./example_files/tide_gauges/l*", date0, date1)

    # TEST: Check length of list
    check1 = len(tidegauge_list) == 2
    # TEST: Check lowestoft matches
    check2 = all(tidegauge_list[1].dataset == lowestoft.dataset)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Multiple tide gauge load")
    else:
        print(str(sec) + chr(subsec) + " X - Multiple tide gauge load")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7j ) Tidegauge map plot (single)                                        #
#                                                                             #
subsec = subsec + 1

# We can take a look at the location of the loaded tidegauge:
try:
    f, a = lowestoft.plot_on_map()
    f.savefig(dn_fig + "tidegauge_map.png")
    print(str(sec) + chr(subsec) + " OK - Tide gauge map plot saved")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

plt.close("all")

# -----------------------------------------------------------------------------#
# %% ( 7k ) Tidegauge map plot (single)                                        #
#                                                                             #
subsec = subsec + 1

# Or we can plot up multiple from the list we loaded:
try:
    f, a = coast.Tidegauge.plot_on_map_multiple(tidegauge_list)
    f.savefig(dn_fig + "tidegauge_multiple_map.png")
    print(str(sec) + chr(subsec) + " OK - Tide gauge multiple map plot saved")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

plt.close("all")

# -----------------------------------------------------------------------------#
# %% ( 7l ) Tidegauge Time series plot                                         #
#                                                                             #
subsec = subsec + 1

# Take a look at the sea level time series stored within the object:

try:
    f, a = lowestoft.plot_timeseries(["sea_level", "sea_level_1H", "sea_level_1H_dx0"])
    f.savefig(dn_fig + "tidegauge_timeseries.png")
    print(str(sec) + chr(subsec) + " OK - Tide gauge time series saved")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

plt.close("all")

# -----------------------------------------------------------------------------#
# %% ( 7m ) Tidegauge method for tabulated data                                #
#                                                                             #
subsec = subsec + 1

# Take a look at the sea level time series stored within the object:
try:
    filnam = "example_files/Gladstone_2020-10_HLW.txt"
    date_start = np.datetime64("2020-10-11 07:59")
    date_end = np.datetime64("2020-10-20 20:21")

    # Initiate a Tidegauge object, if a filename is passed it assumes it is a GESLA type object
    tg = coast.Tidegauge()
    tg.dataset = tg.read_hlw_to_xarray(filnam, date_start, date_end)

    check1 = len(tg.dataset.sea_level) == 37
    check2 = tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="nearest_HW").values == 8.01
    check3 = tg.get_tide_table_times(
        np.datetime64("2020-10-13 12:48"), method="nearest_1"
    ).time.values == np.datetime64("2020-10-13 14:36")
    check4 = np.array_equal(
        tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="nearest_2").values, [2.83, 8.01]
    )
    check5 = np.array_equal(
        tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="window", winsize=24).values,
        [3.47, 7.78, 2.8, 8.01, 2.83, 8.45, 2.08, 8.71],
    )

    if check1 and check2 and check3 and check4 and check5:
        print(str(sec) + chr(subsec) + " OK - Tide table processing")
    else:
        print(str(sec) + chr(subsec) + " X - Tide table processing")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# %% ( 7n ) Tidegauge method for finding extrema and troughs, compare neighbours#
#                                                                             #
subsec = subsec + 1

# Take a look at the sea level time series stored within the object:
try:
    date0 = datetime.datetime(2007, 1, 10)
    date1 = datetime.datetime(2007, 1, 20)
    lowestoft2 = coast.Tidegauge(fn_tidegauge, date_start=date0, date_end=date1)

    # Use comparison of neighbourhood method (method="comp" is assumed)
    extrema_comp = lowestoft2.find_high_and_low_water("sea_level", distance=40)
    # Check actual maximum/minimum is in output dataset
    check1 = np.nanmax(lowestoft2.dataset.sea_level) in extrema_comp.dataset.sea_level_highs
    check2 = np.nanmin(lowestoft2.dataset.sea_level) in extrema_comp.dataset.sea_level_lows
    # Check new time dimensions have correct length (hardcoded here)
    check3 = len(extrema_comp.dataset.time_highs) == 19
    check4 = len(extrema_comp.dataset.time_lows) == 18

    # Attempt a plot
    f = plt.figure()
    plt.plot(lowestoft2.dataset.time, lowestoft2.dataset.sea_level)
    plt.scatter(extrema_comp.dataset.time_highs.values, extrema_comp.dataset.sea_level_highs, marker="o", c="g")
    plt.scatter(extrema_comp.dataset.time_lows.values, extrema_comp.dataset.sea_level_lows, marker="o", c="r")

    plt.legend(["Time Series", "Maxima", "Minima"])
    plt.title("Tide Gauge Optima at Lowestoft")
    f.savefig(dn_fig + "tidegauge_optima.png")

    if check1 and check2 and check3 and check4:
        print(str(sec) + chr(subsec) + " OK - Tidegauge local extrema found")
    else:
        print(str(sec) + chr(subsec) + " X - Tidegauge local extrema")
except:
    print(str(sec) + chr(subsec) + " FAILED.")


# -----------------------------------------------------------------------------#
# %% ( 7o ) Tidegauge method for finding extrema and troughs, fit cubic spline #
#                                                                             #
subsec = subsec + 1

# Load and process BODC processed data
try:
    # Set the start and end dates
    date_start = np.datetime64("2020-10-12 23:59")
    date_end = np.datetime64("2020-10-14 00:01")

    # Initiate a Tidegauge object, if a filename is passed it assumes it is a GESLA
    # type object
    tg = coast.Tidegauge()
    # specify the data read as a High Low Water dataset
    tg.dataset = tg.read_bodc_to_xarray(fn_tidegauge2, date_start, date_end)

    # Use cubic spline fitting method
    extrema_cubc = tg.find_high_and_low_water("sea_level", method="cubic")

    # Check actual maximum/minimum is in output dataset
    check1 = np.isclose(extrema_cubc.dataset.sea_level_highs, [7.77432795, 7.91244559])
    check2 = np.isclose(extrema_cubc.dataset.sea_level_lows, [2.63479458, 2.54599355])

    # Attempt a plot
    f = plt.figure()
    plt.plot(tg.dataset.time, tg.dataset.sea_level)
    plt.scatter(extrema_cubc.dataset.time_highs.values, extrema_cubc.dataset.sea_level_highs, marker="o", c="g")
    plt.scatter(extrema_cubc.dataset.time_lows.values, extrema_cubc.dataset.sea_level_lows, marker="o", c="r")

    plt.legend(["Time Series", "Maxima", "Minima"])
    plt.title("Tide Gauge Optima at Gladstone, fitted cubic spline")
    f.savefig(dn_fig + "tidegauge_optima.png")

    if check1.all() and check2.all():
        print(str(sec) + chr(subsec) + " OK - Tidegauge cubic extrema found")
    else:
        print(str(sec) + chr(subsec) + " X - Tidegauge cubic extrema")
except:
    print(str(sec) + chr(subsec) + " FAILED.")


"""
###############################################################################
## ( 8 ) Isobath Contour Methods                                            ##
###############################################################################
"""
sec = sec + 1
subsec = 96
# -----------------------------------------------------------------------------#
# %% ( 8a ) Extract isbath contour between two points and create contour object  #
#                                                                             #
subsec = subsec + 1
nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_f_grid)
contours, no_contours = coast.Contour.get_contours(nemo_f, 200)
y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_f, contours[0], [50, -10], [60, 3])
cont_f = coast.ContourF(nemo_f, y_ind, x_ind, 200)
if np.isclose(cont_f.y_ind.sum() + cont_f.y_ind.sum(), 190020) and np.isclose(
    cont_f.data_contour.bathymetry.sum().item(), 69803.78125
):
    print(str(sec) + chr(subsec) + " OK - Isobath contour extracted")
else:
    print(str(sec) + chr(subsec) + " X - Isobath contour failed to extract correctly")
# -----------------------------------------------------------------------------#
# %% ( 8b ) Plot contour on map                                                  #
#                                                                             #
subsec = subsec + 1
coast.Contour.plot_contour(nemo_f, contour)
cont_path = dn_fig + "contour.png"
plt.savefig(cont_path)
try:
    if os.path.isfile(cont_path) and os.path.getsize(cont_path) > 0:
        print(str(sec) + chr(subsec) + " OK - Contour plot saved")
    else:
        print(str(sec) + chr(subsec) + " X - Contour plot did not save correctly")
except OSError:
    print(str(sec) + chr(subsec) + " X - Contour plot did not save correctly")
# -----------------------------------------------------------------------------#
# %% ( 8c ) Calculate pressure along contour                                     #
#                                                                             #
subsec = subsec + 1
nemo_t = coast.Gridded(fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid)
contours, no_contours = coast.Contour.get_contours(nemo_t, 200)
y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_t, contours[0], [50, -10], [60, 3])
cont_t = coast.ContourT(nemo_t, y_ind, x_ind, 200)
cont_t.construct_pressure(1027)
if np.allclose(
    (cont_t.data_contour.pressure_s + cont_t.data_contour.pressure_h_zlevels).sum().compute().item(), 27490693.20181531
):
    print(str(sec) + chr(subsec) + " OK - Perturbation pressure calculation is as expected")
else:
    print(str(sec) + chr(subsec) + " X - Perturbation pressure calculation is not as expected")
# -----------------------------------------------------------------------------#
# %% ( 8d ) Calculate flow across contour                                        #
#                                                                             #
subsec = subsec + 1
nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=fn_config_f_grid)
nemo_u = coast.Gridded(fn_data=dn_files + fn_nemo_grid_u_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_u_grid)
nemo_v = coast.Gridded(fn_data=dn_files + fn_nemo_grid_v_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_v_grid)
contours, no_contours = coast.Contour.get_contours(nemo_f, 200)
y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_f, contours[0], [50, -10], [60, 3])
cont_f = coast.ContourF(nemo_f, y_ind, x_ind, 200)
cont_f.calc_cross_contour_flow(nemo_u, nemo_v)
if np.allclose(
    (cont_f.data_cross_flow.normal_velocities + cont_f.data_cross_flow.depth_integrated_normal_transport).sum(),
    -1152.3771,
):
    print(str(sec) + chr(subsec) + " OK - Cross-contour flow calculations as expected")
else:
    print(str(sec) + chr(subsec) + " X - Cross-contour flow calculations not as expected")
# -----------------------------------------------------------------------------#
# %% ( 8e ) Calculate pressure gradient driven flow across contour               #
#                                                                             #
subsec = subsec + 1
cont_f.calc_geostrophic_flow(nemo_t, config_u=fn_config_u_grid, config_v=fn_config_v_grid, ref_density=1027)
if np.allclose(
    (
        cont_f.data_cross_flow.normal_velocity_hpg
        + cont_f.data_cross_flow.normal_velocity_spg
        + cont_f.data_cross_flow.transport_across_AB_hpg
        + cont_f.data_cross_flow.transport_across_AB_spg
    ).sum(),
    74.65002414,
):
    print(str(sec) + chr(subsec) + " OK - Cross-contour geostrophic flow calculations as expected")
else:
    print(str(sec) + chr(subsec) + " X - Cross-contour geostrophic flow calculations not as expected")

# %%
"""
###############################################################################
## ( 9 ) EOF module testing                                                 ##
###############################################################################
"""
sec = sec + 1
subsec = 96

# %%---------------------------------------------------------------------------#
# ( 9a ) Compute regular EOFs, temporal projections and variance explained   #
#
subsec = subsec + 1
try:
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )
    eofs = coast.compute_eofs(nemo_t.dataset.ssh)

    ssh_reconstruction = (eofs.EOF * eofs.temporal_proj).sum(dim="mode").sum(dim=["x_dim", "y_dim"])
    ssh_anom = (nemo_t.dataset.ssh - nemo_t.dataset.ssh.mean(dim="t_dim")).sum(dim=["x_dim", "y_dim"])

    # Check ssh anomaly is reconstructed at each time point
    if np.allclose(ssh_reconstruction, ssh_anom, rtol=0.0001):
        var_cksum = eofs.variance.sum(dim="mode").compute().item()
        if np.isclose(var_cksum, 100):
            print(str(sec) + chr(subsec) + " OK - Original signal reconstructed from EOFs")
        else:
            print(str(sec) + chr(subsec) + " X - Variance explained does not sum to 100 %")
    else:
        print(str(sec) + chr(subsec) + " X - Original signal not reconstructed from EOFs")
except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

# %%---------------------------------------------------------------------------#
# ( 9b ) Compute  HEOFs, temporal projections and variance explained   #
#
subsec = subsec + 1
try:
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )
    heofs = coast.compute_hilbert_eofs(nemo_t.dataset.ssh)

    ssh_reconstruction = (
        (heofs.EOF_amp * heofs.temporal_amp * uf.exp(1j * uf.radians(heofs.EOF_phase + heofs.temporal_phase)))
        .sum(dim="mode")
        .real.sum(dim=["x_dim", "y_dim"])
    )

    ssh_anom = (nemo_t.dataset.ssh - nemo_t.dataset.ssh.mean(dim="t_dim")).sum(dim=["x_dim", "y_dim"])

    # Check ssh anomaly is reconstructed at each time point
    if np.allclose(ssh_reconstruction, ssh_anom, rtol=0.0001):
        var_cksum = heofs.variance.sum(dim="mode").item()
        if np.isclose(var_cksum, 100):
            print(str(sec) + chr(subsec) + " OK - Original signal reconstructed from HEOFs")
        else:
            print(str(sec) + chr(subsec) + " X - Variance explained does not sum to 100 %")
    else:
        print(str(sec) + chr(subsec) + " X - Original signal not reconstructed from HEOFs")


except:
    print(str(sec) + chr(subsec) + " FAILED.\n" + traceback.format_exc())

"""
#################################################
## ( 10 ) Profile Methods                     ##
#################################################
"""
sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# ( 10a ) Load EN4 data                                                       #
#                                                                             #
subsec = subsec + 1
# <Introduction>

try:
    profile = coast.Profile(fn_profile, config=fn_profile_config)
    profile.dataset = profile.dataset.isel(profile=np.arange(0, profile.dataset.dims["profile"], 10)).load()

    check1 = type(profile) == coast.Profile
    check2 = profile.dataset.temperature.values[0, 0] = 8.981

    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10b ) Process EN4 data                                                     #
#
subsec = subsec + 1
# <Introduction>

try:
    processed = profile.process_en4()
    processed.dataset.load()

    check1 = type(processed) == coast.profile.Profile
    check2 = np.isnan(processed.dataset.temperature.values[0, 0])
    check3 = processed.dataset.dims["profile"] == 111

    if check1 and check2 and check3:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10c ) Gridded obs_operaor                                                  #
#
subsec = subsec + 1
# <Introduction>

try:
    # First read some new Gridded data
    nemo_t = coast.Gridded(
        fn_data=dn_files + fn_nemo_grid_t_dat, fn_domain=dn_files + fn_nemo_dom, config=fn_config_t_grid
    )
    nemo_t.dataset["landmask"] = nemo_t.dataset.bottom_level == 0
    nemo_profiles = processed.obs_operator(nemo_t)

    check1 = type(nemo_profiles) == coast.profile.Profile
    check2 = "nearest_index_x" in list(nemo_profiles.dataset.keys())
    check3 = nemo_profiles.dataset.interp_dist.values[0] == 151.4443554515237

    if check1 and check2 and check3:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10d ) Vertical Interpolation                                               #
#
subsec = subsec + 1
# <Introduction>

try:
    reference_depths = np.arange(0, 500, 2)
    nemo_profiles.dataset = nemo_profiles.dataset.rename({"depth_0": "depth"})
    model_interpolated = nemo_profiles.interpolate_vertical(processed)

    check1 = type(model_interpolated) == coast.profile.Profile
    check2 = nemo_profiles.dataset.temperature.values[0, 0] == np.float32(1.7324219)

    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10e ) Profile Differencing                                                 #
#
subsec = subsec + 1
# <Introduction>

try:
    difference = processed.difference(model_interpolated)
    difference.dataset.load()

    check1 = type(difference) == coast.profile.Profile
    check2 = difference.dataset.diff_temperature.values[0, 2] == np.float32(1.1402345)

    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10f ) Regional Averaging                                                   #
#
subsec = subsec + 1
# <Introduction>

try:
    # First let's make some masks to define regions of the model domain
    mm = coast.MaskMaker()

    # Make some variables easier to access
    bath = nemo_t.dataset.bathymetry.values
    lon = nemo_t.dataset.longitude.values
    lat = nemo_t.dataset.latitude.values

    mm_north_sea = mm.region_def_nws_north_sea(lon, lat, bath)
    mm_whole_domain = np.ones(lon.shape)
    mask_list = [mm_north_sea, mm_whole_domain]
    mask_names = ["North Sea", "Whole Domain"]

    # Turn mask list into an xarray dataset
    mask_list = coast.MaskMaker.make_mask_dataset(lon, lat, mask_list)

    # Determine whether each profile is in each masked region or not
    mask_indices = model_interpolated.determine_mask_indices(mask_list)

    # Do average differences for each region
    mask_means = difference.mask_means(mask_indices)

    check1 = mask_means.average_diff_temperature.values[0] == np.float32(-0.78869253)

    if check1:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 10g ) Surface/Bottom Averaging                                             #
#
subsec = subsec + 1
# <Introduction>

try:

    # Lets get surface values by averaging over the top 5m of data
    surface = 5
    model_profiles_surface = nemo_profiles.depth_means([0, surface])

    # Lets get bottom values by averaging over the bottom 30m, except whether
    # depth is <100m, then average over the bottom 10m
    model_profiles_bottom = nemo_profiles.bottom_means([10, 30], [100, np.inf])

    check1 = type(model_profiles_surface) == coast.profile.Profile
    check1 = type(model_profiles_bottom) == coast.profile.Profile
    check3 = model_profiles_surface.dataset.temperature.values[0] == np.float32(1.7500391)

    if check1 and check2 and check3:
        print(str(sec) + chr(subsec) + " OK")
    else:
        print(str(sec) + chr(subsec) + " X")

except:
    print(str(sec) + chr(subsec) + " FAILED.")
# %%
"""
#################################################
## ( 11 ) PLOTTING UTILITY Methods             ##
#################################################
"""
sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# ( 11a ) Scatter with fit                                                     #
#                                                                             #

subsec = subsec + 1
# <Introduction>

try:
    # Plot an idealised dataset
    x = np.arange(0, 50)
    y = np.arange(0, 50) / 1.5
    f, a = plot_util.scatter_with_fit(x, y)
    a.set_title("Test: Scatter_with_fit()")

    f.savefig(os.path.join(dn_fig, "scatter_with_fit_test.png"))
    plt.close()

    print(str(sec) + chr(subsec) + " OK. Scatter_with_fit()")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 11b ) Geo axes                                                             #
#                                                                             #

subsec = subsec + 1

try:
    # Plot two scatters on a map
    lonbounds = [-20, 20]
    latbounds = [30, 60]
    f, a = plot_util.create_geo_axes(lonbounds, latbounds)
    a.set_title("Test: create_geo_axes()")
    a.scatter([0, -10], [50, 50])

    f.savefig(os.path.join(dn_fig, "create_geo_axes_test.png"))
    plt.close()

    print(str(sec) + chr(subsec) + " OK. create_geo_axes()")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 11c ) Determine colorbar extension                                         #
#                                                                             #

subsec = subsec + 1

try:
    # Test on some pretend dataset
    pretend_data = np.arange(0, 50)
    test1 = plot_util.determine_colorbar_extension(pretend_data, -50, 100)
    test2 = plot_util.determine_colorbar_extension(pretend_data, 1, 100)
    test3 = plot_util.determine_colorbar_extension(pretend_data, -50, 48)
    test4 = plot_util.determine_colorbar_extension(pretend_data, 1, 48)

    # TEST: <description here>
    check1 = test1 == "neither"
    check2 = test2 == "min"
    check3 = test3 == "max"
    check4 = test4 == "both"
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - determine_colorbar_extension()")
    else:
        print(str(sec) + chr(subsec) + " X - ")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 11d ) Determine clim by st.dev                                             #
#                                                                             #

subsec = subsec + 1
# <Introduction>

try:
    # Data with mean = 50.51 and std = 32.2
    # Determine clims to exclude outlier at 200
    pretend_data = np.arange(0, 100)
    pretend_data[-1] = 200
    clim = plot_util.determine_clim_by_standard_deviation(pretend_data, n_std_dev=2)

    # TEST: <description here>
    check1 = clim[0] == -13.808889915793792
    check2 = clim[1] == 114.82888991579378
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - determine_clim_by_std_dev")
    else:
        print(str(sec) + chr(subsec) + " X - determine_clim_by_std_dev")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %%
"""
#################################################
## ( 12 ) Stats Utility                        ##
#################################################
"""

sec = sec + 1
subsec = 96

# -----------------------------------------------------------------------------#
# ( 12a ) find_maxima(). Test comparison and cublic spline methods            #
#                                                                             #

subsec = subsec + 1


try:
    date0 = datetime.datetime(2007, 1, 15)
    date1 = datetime.datetime(2007, 1, 16)
    tg = coast.Tidegauge(fn_tidegauge, date_start=date0, date_end=date1)

    tt, hh = stats_util.find_maxima(tg.dataset.time, tg.dataset.sea_level, method="comp")
    check1 = np.isclose((tt.values[0] - np.datetime64("2007-01-15T00:15:00")) / np.timedelta64(1, "s"), 0)
    check2 = np.isclose(hh.values[0], 1.027)

    tt, hh = stats_util.find_maxima(tg.dataset.time, tg.dataset.sea_level, method="cubic")
    check3 = np.isclose((tt[0] - np.datetime64("2007-01-15T00:07:49")) / np.timedelta64(1, "s"), 0)
    check4 = np.isclose(hh[0], 1.0347638302097757)

    if check1 and check2 and check3 and check4:
        print(str(sec) + chr(subsec) + " OK - find_maxima() worked for comparison and cublic spline methods")
    else:
        print(str(sec) + chr(subsec) + " X - Problem with stats_util.find_maxima()")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %%
"""
#################################################
## ( 13 ) MaskMaker                           ##
#################################################
"""

sec = sec + 1
subsec = 96

# Preparation: Create two arrays to put mask onto, one of zeros and one of ones
# This allows us to test the additive feature.
sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)
mask00 = np.zeros((sci.dataset.dims["y_dim"], sci.dataset.dims["x_dim"]))
mask01 = np.ones((sci.dataset.dims["y_dim"], sci.dataset.dims["x_dim"]))

# -----------------------------------------------------------------------------#
# ( 13a ) Create mask by indices                                              #
#                                                                             #

subsec = subsec + 1
# Plot ts diagram

try:
    mm = coast.MaskMaker()
    # Draw and fill a square
    vertices_r = [50, 150, 150, 50]
    vertices_c = [50, 50, 150, 150]
    filled0 = mm.fill_polygon_by_index(mask00, vertices_r, vertices_c)
    filled1 = mm.fill_polygon_by_index(mask01, vertices_r, vertices_c, additive=True)

    # TEST: Check some data
    check1 = filled0[49, 49] == 0 and filled0[51, 51] == 1
    check2 = filled1[49, 49] == 1 and filled1[51, 51] == 2
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - MASKS created by index")
    else:
        print(str(sec) + chr(subsec) + " X - Problem mask creation by index")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# -----------------------------------------------------------------------------#
# ( 13b ) Create mask by lonlat                                               #
#                                                                             #

subsec = subsec + 1
# Plot ts diagram

try:
    mm = coast.MaskMaker()
    # Draw and fill a square
    vertices_lon = [-5, -5, 5, 5]
    vertices_lat = [40, 60, 60, 40]
    filled0 = mm.fill_polygon_by_lonlat(mask00, sci.dataset.longitude, sci.dataset.latitude, vertices_lon, vertices_lat)
    filled1 = mm.fill_polygon_by_lonlat(
        mask01, sci.dataset.longitude, sci.dataset.latitude, vertices_lon, vertices_lat, additive=True
    )

    # TEST: Check some data
    check1 = filled0[50, 50] == 0 and filled0[50, 150] == 1
    check2 = filled1[50, 50] == 1 and filled1[50, 150] == 2
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - MASKS created by lonlat")
    else:
        print(str(sec) + chr(subsec) + " X - Problem mask creation by lonlat")

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %%
"""
#################################################
## ( 14 ) Climatology Methods                  ##
#################################################
"""
sec = sec + 1
subsec = 96

sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)
ds = sci.dataset[["temperature", "ssh"]].isel(z_dim=0)


# -----------------------------------------------------------------------------#
# ( 14a ) Monthly and Seasonal Climatology                                     #
#                                                                             #

subsec = subsec + 1

try:

    clim = coast.Climatology()
    fn_out = os.path.join(dn_files, "test_climatology.nc")
    monthly = clim.make_climatology(ds, "month").load()
    seasonal = clim.make_climatology(ds, "season", fn_out=fn_out)

    # create dataset with missing values
    ds2 = ds.copy(deep=True)
    ds2["temperature"][::2, :100, :100] = np.nan
    ds2["ssh"][::2, :100, :100] = np.nan
    seaC = clim.make_climatology(ds2, "season")
    seaX = ds2.groupby("time.season").mean("t_dim")
    # throws error is not close
    xr.testing.assert_allclose(seaC, seaX)

    mn = mn = np.nanmean(ds.temperature, axis=0)
    check1 = np.nanmax(np.abs(mn - monthly.temperature[0])) < 1e-6
    check2 = os.path.isfile(fn_out)
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - Monthly and seasonal climatology made and written to file")
    else:
        print(str(sec) + chr(subsec) + " X - Problem with monthly and seasonal climatology ")

except AssertionError:
    print(str(sec) + chr(subsec) + " X - Problem with computing climatology when dataset has missing values")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %%
"""
#################################################
## ( 15 ) XESMF Interface                      ##
#################################################
"""
sec = sec + 1
subsec = 96

sci = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, config=fn_config_t_grid)

#%%
# -----------------------------------------------------------------------------#
# ( 15a ) Passing a single gridded object                                     #
#                                                                             #

subsec = subsec + 1

try:

    xesmf_ready = xs = coast.xesmf_convert(sci)
    check_grid = xesmf_ready.input_grid
    check_data = xesmf_ready.input_data
    check1 = np.array_equal(check_grid.lat.values, sci.dataset.latitude.values )
    check2 = np.array_equal(check_data.temperature[0,0].values, 
                            sci.dataset.temperature[0,0].values, equal_nan=True)
    
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - xesmf_convert with single gridded obj")
    else:
        print(str(sec) + chr(subsec) + " X - xesmf_convert with single gridded obj ")

except AssertionError:
    print(str(sec) + chr(subsec) + " X - xesmf_convert with single gridded obj")
except:
    print(str(sec) + chr(subsec) + " FAILED.")
    
#%%
# -----------------------------------------------------------------------------#
# ( 15b ) Passing two gridded object                                          #
#                                                                             #

subsec = subsec + 1

try:

    xesmf_ready = xs = coast.xesmf_convert(sci, sci)
    check_grid0 = xesmf_ready.input_grid
    check_grid1 = xesmf_ready.input_data
    check1 = np.array_equal(check_grid0.lat.values, sci.dataset.latitude.values )
    check2 = np.array_equal(check_grid1.lat.values, sci.dataset.latitude.values )
    
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - xesmf_convert with single gridded obj")
    else:
        print(str(sec) + chr(subsec) + " X - xesmf_convert with single gridded obj ")

except AssertionError:
    print(str(sec) + chr(subsec) + " X - xesmf_convert with single gridded obj")
except:
    print(str(sec) + chr(subsec) + " FAILED.")
    
#%%
# -----------------------------------------------------------------------------#
# ( 15c ) Convert back to gridded object                                       #
#                                                                             #

subsec = subsec + 1

try:

    xesmf_ready = xs = coast.xesmf_convert(sci, sci)
    check_grid0 = xesmf_ready.input_grid
    check_grid1 = xesmf_ready.output_grid
    gridded_again = xesmf_ready.to_gridded(xesmf_ready.input_data)
    check1 = np.array_equal(sci.dataset.latitude, gridded_again.dataset.latitude)
    check2 = np.array_equal(sci.dataset.temperature[0,0], gridded_again.dataset.temperature[0,0],
                            equal_nan=True)
    
    if check1 and check2:
        print(str(sec) + chr(subsec) + " OK - xesmf_convert to_gridded()")
    else:
        print(str(sec) + chr(subsec) + " X - xesmf_convert to_gridded() ")

except AssertionError:
    print(str(sec) + chr(subsec) + " X - xesmf_convert to_gridded()")
except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %%
"""
###############################################################################
## ( N ) Example script testing                                              ##
###############################################################################
"""
sec = "N"
subsec = 96

print(str(sec) + ". Example script testing")
print("++++++++++++++++++++++++")
print("  script output follows  ")
print(" ")
#
# -----------------------------------------------------------------------------#
# %% ( Na ) Example script testing                                               #
#                                                                             #
subsec = subsec + 1
# Test machine name (to check for file access) in order to test additional scripts.
example_script_flag = True if "livljobs" in gethostname().lower() else False

try:
    # Do a thing
    from example_scripts import altimetry_tutorial  # This runs on example_files
    from example_scripts import tidegauge_tutorial  # This runs on example_files
    from example_scripts import tidetable_tutorial  # This runs on example_files
    from example_scripts import export_to_netcdf_tutorial  # This runs on example_files
    from example_scripts import transect_tutorial  # This runs on example_files
    from example_scripts import contour_tutorial  # This runs on example_files
    from example_scripts import (
        internal_tide_pycnocline_diagnostics,
    )  # This runs on example_files unless it is on livljobs, then it is AMM60 data

    print(" ")
    print("  script output ends  ")
    print("++++++++++++++++++++++++")
    print(str(sec) + chr(subsec) + " OK - tutorials on example_files data")
    subsec = subsec + 1

    if example_script_flag:
        from example_scripts import amm15_example_plot

        print(str(sec) + chr(subsec) + " OK - tutorial on AMM15 data")

        subsec = subsec + 1
        from example_scripts import anchor_plots_of_nsea_wvel

        print(str(sec) + chr(subsec) + " OK - tutorial on AMM60 data")

        subsec = subsec + 1
        from example_scripts import blz_example_plot

        print(str(sec) + chr(subsec) + " OK - tutorial on Belize data")

        subsec = subsec + 1
        from example_scripts import seasia_r12_example_plot

        print(str(sec) + chr(subsec) + " OK - tutorial on SEAsia data")

        subsec = subsec + 1
        from example_scripts import wcssp_india_example_plot

        print(str(sec) + chr(subsec) + " OK - tutorial on WCSSP-India data")

        subsec = subsec + 1  # ON LIVLJOBS THIS RUNS TWICE FOR AMM60 DATA.
        from example_scripts import internal_tide_pycnocline_diagnostics

        print(str(sec) + chr(subsec) + " OK - tutorial on internal tides")
    else:
        print("Don't forget to test on a LIVLJOBS machine")

    # TEST: <description here>
    check1 = example_script_flag
    if check1:
        print(str(sec) + " OK - example_scripts ran on", gethostname())
    else:
        print(str(sec) + " X - example_scripts failed on", gethostname())

except:
    print(str(sec) + chr(subsec) + " FAILED.")

# %% Close log file
#################################################
log_file.close()
