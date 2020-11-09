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
nemo = coast.NEMO(fn_detided , fn_domain, grid_ref = 't-grid')
#nemo = nemo.isel(x_dim = np.arange(0,1760,1), y_dim = np.arange(0,1100,1))

# Merge together all the different sources of altimetry.
step=10
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
radii = np.arange(100,2000,100)
obs_var = 'sla_unfiltered'

ctr_lon = nemo.dataset.longitude.values[::20,::20]
ctr_lat = nemo.dataset.latitude.values[::20,::20]
ctr_mask = ssh.values[0,::20, ::20]
ctr_mask = ~np.isnan(ctr_mask)
ctr_lon = ctr_lon[ctr_mask]
ctr_lat = ctr_lat[ctr_mask]

# Obs operator
print('obs_operator')
alt.obs_operator(ssh, time_interp='linear')

alt.gradient_alongtrack('interp_ssh')
alt.gradient_alongtrack('sla_unfiltered')

vals = []

print('Starting loop over radii')
for rr in radii:
    
    d_ind = gu.subset_indices_by_distance_BT(alt_lon.values, alt_lat.values, 
                                             ctr_lon, ctr_lat, rr)
    print(rr)
    tmp = np.zeros(len(ctr_lon))        
    
    for ii in range(0,len(ctr_lon)):
        asub = alt.dataset['grad_sla_unfiltered'].values
        asub = asub[d_ind[ii]]
        msub = alt.dataset['grad_interp_ssh'].values
        msub = msub[d_ind[ii]]
        
        acdf = coast.CDF(asub)
        mcdf = coast.CDF(msub)
        
        tmp[ii] = mcdf.integral(acdf)
    vals.append(np.array(tmp))