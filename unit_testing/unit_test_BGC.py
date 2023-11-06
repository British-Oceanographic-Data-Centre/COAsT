##
"""
Script to do unit testing for BGC only in SEAsia
***SECTIONS***
This script is separated into Subsections, for which there are two
counters to keep track: sec and subsec respectively.  At the beginning of each
section, the sec counter should be incremented by 1 and the subsec counter
should be reset to 96 (code for one below 'a'). At the beginning of each
subsection, subsec should be incremented by one.
***OTHER FILES***
This is supplement file to the original unit testing script:
    - unit_test
Run:
ipython: cd COAsT; run unit_testing/unit_test_BGC.py  # I.e. from the git repo.
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

plt.switch_backend("agg")
import datetime
import os.path as path
import logging
import coast._utils.general_utils as general_utils
import coast._utils.plot_util as plot_util
import coast._utils.stats_util as stats_util
from socket import gethostname  # to get hostname
import traceback

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
# BGC for nemo
fn_nemo_dat = "coast_example_SEAsia_BGC_1990.nc"
fn_nemo_dom_bgc = "coast_example_domain_SEAsia.nc"
# BGC for SEAsia
fn_nemo_config_bgc_grid = path.join("./config", "example_nemo_bgc.json")

sec = 1
subsec = 96  # Code for '`' (1 below 'a')
"""
#################################################
## ( 1 ) Loading/Initialisation                ##
#################################################
"""
# This section is for testing the loading and initialisation of Gridded objects.

# -----------------------------------------------------------------------------#
# %% ( 1a ) BGC- Load example Gridded Nemo-nemo data for BGC (DIC, oxygen, nutrients, ph, alkalinity)                 #
#                                                                             #

subsec = subsec + 1

try:
    sci_bgc = coast.Gridded(
        path.join(dn_files, fn_nemo_dat), path.join(dn_files, fn_nemo_dom_bgc), config=fn_nemo_config_bgc_grid
    )

    # Test the data has loaded
    sci_bgc_attrs_ref = dict(
        [
            ("name", "SEAsia_HAD_1m_19900101_19901231_ptrc_T"),
            ("description", "tracer variables"),
            ("title", "tracer variables"),
            ("Conventions", "CF-1.6"),
            ("timeStamp", "2020-Oct-07 10:11:58 GMT"),
            ("uuid", "701bb916-558d-4ee8-9cf6-89454c7bc99f"),
        ]
    )

    # checking is LHS is a subset of RHS
    if sci_bgc_attrs_ref.items() <= sci_bgc.dataset.attrs.items():
        print(str(sec) + chr(subsec) + " OK - nemo BGC data loaded: " + fn_nemo_dat)
    else:
        print(str(sec) + chr(subsec) + " X - There is an issue with loading nemo BGC " + fn_nemo_dat)
except:
    print(str(sec) + chr(subsec) + " FAILED BGC")

# -----------------------------------------------------------------------------#
# %% ( 1b ) BGC Load data from existing dataset                                      #
#                                                                             #

subsec = subsec + 1

try:
    ds_bgc = xr.open_dataset(dn_files + fn_nemo_dat)
    sci_bgc_load_ds = coast.Gridded(config=fn_nemo_config_bgc_grid)
    sci_bgc_load_ds.load_dataset(ds_bgc)
    sci_bgc_load_file = coast.Gridded(config=fn_nemo_config_bgc_grid)
    sci_bgc_load_file.load(dn_files + fn_nemo_dat)
    if sci_bgc_load_ds.dataset.identical(sci_bgc_load_file.dataset):
        print(str(sec) + chr(subsec) + " OK bgc - coast.load_dataset()")
    else:
        print(
            str(sec)
            + chr(subsec)
            + " X BGC - coast.load_dataset() ERROR - not identical to dataset loaded via coast.load()"
        )
except:
    print(str(sec) + chr(subsec) + " FAILED BGC")


# -----------------------------------------------------------------------------#
# %% ( 1c )  BGC Set Gridded variable name                                               #
#                                                                             #

subsec = subsec + 1

try:
    sci_bgc = coast.Gridded(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom_bgc, config=fn_nemo_config_bgc_grid)
    try:
        sci_bgc.dataset.DIC
    except NameError:
        print(str(sec) + chr(subsec) + " X - variable name (to DIC) not reset")
    else:
        print(str(sec) + chr(subsec) + " OK - variable name reset (to DIC)")
except:
    print(str(sec) + chr(subsec) + " FAILED BGC")

# -----------------------------------------------------------------------------#
# %% ( 1d ) BGC Set Gridded grid attributes - dimension names                           #
#                                                                             #

subsec = subsec + 1  #

try:
    if sci_bgc.dataset.DIC.dims == ("t_dim", "z_dim", "y_dim", "x_dim"):
        print(str(sec) + chr(subsec) + " OK BGC - dimension names reset")
    else:
        print(str(sec) + chr(subsec) + " X BGC - dimension names not reset")
except:
    print(str(sec) + chr(subsec) + " FAILED BCG")


# -----------------------------------------------------------------------------#
# %% ( 1e ) BGC Load only domain data in Gridded                                        #
#                                                                             #

subsec = subsec + 1

pass_test = False
nemo_f = coast.Gridded(fn_domain=dn_files + fn_nemo_dom_bgc, config=fn_nemo_config_bgc_grid)

if nemo_f.dataset._coord_names == {"depth_0", "latitude", "longitude"}:
    var_name_list = []
    for var_name in nemo_f.dataset.data_vars:
        var_name_list.append(var_name)
    if var_name_list == ["bathymetry", "e1", "e2", "e3_0", "bottom_level"]:
        pass_test = True

if pass_test:
    print(str(sec) + chr(subsec) + " OK BGC - Gridded loaded domain data only")
else:
    print(str(sec) + chr(subsec) + " X BGC - Gridded didn't load domain data correctly")

# -----------------------------------------------------------------------------#
# %% ( 1f ) Plot surface DIC testing                                              #
#                                                                             #

subsec = subsec + 1
try:
    fig = plt.figure()
    plt.pcolormesh(
        sci_bgc.dataset.longitude,
        sci_bgc.dataset.latitude,
        sci_bgc.dataset.DIC.isel(t_dim=0).isel(z_dim=0),
        cmap="RdYlBu_r",
        vmin=1600,
        vmax=2080,
    )
    plt.colorbar()
    plt.title("DIC, mmol/m^3")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()
    fig.savefig("SEAsia_DIC_surface.png")
    print(str(sec) + chr(subsec) + " OK - DIC at surface plot saved")
except:
    print(str(sec) + chr(subsec) + "X - DIC plot failed")


# %% Close log file
#################################################
log_file.close()
