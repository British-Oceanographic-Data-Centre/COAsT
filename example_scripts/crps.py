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
import numpy as np


dir = 'example_files/'
fn_dom = dir + 'COAsT_example_NEMO_domain.nc'
fn_dat = dir + 'COAsT_example_NEMO_data.nc'
fn_alt = dir + 'COAsT_example_altimetry_data.nc'

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
