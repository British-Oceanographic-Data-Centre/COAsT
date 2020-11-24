# Run this script for BOTH storm events and both tided/detided data
#
# Also need to show radius sizes

import coast
import coast.general_utils as gu
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Paths
fn_detided = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/detided.nc'
fn_domain = '/Users/Dave/Documents/Projects/WCSSP/Data/domain_cfg_wcssp.nc'
fn_alt_list = ['/Users/Dave/Documents/Projects/WCSSP/Data/fani/*j3*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*s3a*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*s3b*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*h2g*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*c2*']
                  

# Load NEMO data and thin it out a bit for speed/memory
nemo = coast.NEMO(fn_detided , fn_domain, grid_ref = 't-grid', chunks={})
#nemo = nemo.isel(x_dim = np.arange(0,1760,3), y_dim = np.arange(0,1100,3))

# Merge together all the different sources of altimetry.
step=25
lon_bounds = (65,99)
lat_bounds = (3.5, 27)
alt = coast.ALTIMETRY()
alt.dataset = xr.Dataset()
for fn_alt in fn_alt_list:
    alt_tmp = coast.ALTIMETRY(fn_alt, multiple=True)
    ind = alt_tmp.subset_indices_lonlat_box(lon_bounds,lat_bounds)
    alt_tmp = alt_tmp.isel(time=ind[::step])
    alt.dataset = xr.merge((alt_tmp.dataset, alt.dataset))
alt.dataset = alt.dataset.rename_dims({'time':'t_dim'})

print('Altimetry merged')
# Remove Dynamic Atmospheric Correction from the data (?)
#alt.dataset['no_dac'] = alt.dataset.sla_unfiltered + alt.dataset.dac

# Define some variable references for ease
alt_lon = alt.dataset.longitude
alt_lat = alt.dataset.latitude
alt_time = alt.dataset.time
ssh = nemo.dataset.ssh
ssh.load()
mlon = ssh.longitude
mlat = ssh.latitude
radii = np.arange(5,50,2)
obs_var = 'sla_unfiltered'

crps_vals = []
crps_land = []

print('Starting loop over radii')
for rr in radii:
    
    print(rr)
    #### CRPS
    crps = alt.crps(ssh, obs_var, rr)
    crps_vals.append( np.array(crps.dataset.crps.values))
    crps_land.append( np.array(crps.dataset.crps_contains_land.values))
    print('crps done')