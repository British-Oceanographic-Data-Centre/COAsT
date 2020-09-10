# CRPS.py
"""
Script to demonstrate Continuosly Ranked Probability Score functionality using
 the COAsT package.

This forms the basis of the html exmaple at:
 https://british-oceanographic-data-centre.github.io/COAsT/

This is a basic script for running the CRPS function with the example NEMO data
 and Altimetry data. Altimetry data currently being read in using netCDF4 and
 cut out of global domain before being given to the routine.
"""

import coast
import os
import numpy as np


example_dir = 'example_files/'
if not os.path.isdir(example_dir):
    print("please go download the examples file from https://dev.linkedsystems.uk/erddap/files/COAsT_example_files/")
    example_dir = input("what is the path to the example files:\n")
    if not os.path.isdir(example_dir):
        print(f"location f{example_dir} cannot be found")



fn_dom = example_dir + 'COAsT_example_NEMO_domain.nc'
fn_dat = example_dir + 'COAsT_example_NEMO_data.nc'
fn_alt = example_dir + 'COAsT_example_altimetry_data.nc'

nemo_dom = coast.DOMAIN()
nemo_var = coast.NEMO()
altimetry = coast.ALTIMETRY()

nemo_dom.load(fn_dom)
nemo_var.load(fn_dat)
altimetry.load(fn_alt)

altimetry.set_command_variables()
nemo_var.set_command_variables()
nemo_dom.set_command_variables()



ind = altimetry.subset_indices_lonlat_box([-10,10], [45,60])
altimetry_nwes = altimetry.isel(time=ind) #nwes = northwest europe shelf


alt_tmp = altimetry_nwes.subset_as_copy(time=[0,1,2,3,4])
crps_rad = nemo_var.crps_sonf('sossheig', nemo_dom, alt_tmp, 'sla_filtered',
                    nh_radius=111, nh_type = "radius", cdf_type = "empirical",
                    time_interp = "nearest", plot=True)
crps_box = nemo_var.crps_sonf('sossheig', nemo_dom, alt_tmp, 'sla_filtered',
                    nh_radius=1, nh_type = "box", cdf_type = "theoretical",
                    time_interp = "nearest", plot=False)
