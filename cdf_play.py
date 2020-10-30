# Run this script for BOTH storm events and both tided/detided data
#
# Also need to show radius sizes

import coast
import coast.general_utils as gu
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

fn_detided = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/detided.nc'
fn_domain = '/Users/Dave/Documents/Projects/WCSSP/Data/domain_cfg_wcssp.nc'
fn_alt_list = ['/Users/Dave/Documents/Projects/WCSSP/Data/fani/*j3*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*s3a*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*s3b*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*h2g*',
               '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*c2*']
                  

nemo = coast.NEMO(fn_detided , fn_domain, grid_ref = 't-grid')
nemo = nemo.isel(x_dim = np.arange(0,1760,2), y_dim = np.arange(0,1100,2))

step=15
lon_bounds = (65,99)
lat_bounds = (3.5, 27)
alt = coast.ALTIMETRY()
alt.dataset = xr.Dataset()
for fn_alt in fn_alt_list:
    alt_tmp = coast.ALTIMETRY(fn_alt, multiple=True)
    ind = alt_tmp.subset_indices_lonlat_box(lon_bounds,lat_bounds)
    alt_tmp = alt_tmp.isel(time=ind[::step])
    alt.dataset = xr.merge((alt_tmp.dataset, alt.dataset))

#alt.dataset['no_dac'] = alt.dataset.sla_unfiltered + alt.dataset.dac

alon = alt.dataset.longitude
alat = alt.dataset.latitude
lon0 = alt.dataset.longitude[300]
lat0 = alt.dataset.latitude[300]
time0 = alt.dataset.time[300]
ssh = nemo.dataset.ssh
ssh.load()
mlon = ssh.longitude
mlat = ssh.latitude
radii = np.arange(10,200,10)

# Obs operator
print('obs_operator')
alt.obs_operator(ssh, time_interp='linear')

crps_mean = []
crps_mean_noland = []

for rr in radii:
    
    #### CRPS
    crps = alt.crps(ssh, 'sla_unfiltered', 10)

    #### > 2D Model Sample Vs Along Track Sample: tva
    x,y = gu.subset_indices_by_distance(mlon, mlat, lon0, lat0, radius)
    msub = ssh.isel(x_dim=x, y_dim=y)
    msub = gu.interpolate_in_time(msub, time0)
    t = gu.subset_indices_by_distance(alon, alat, lon0, lat0, radius)
    asub = alt.isel(t_dim=t)
    mcdf = coast.CDF(msub.values)
    acdf = coast.CDF(asub.dataset.sla_unfiltered.values)
    
    #### > Interpolated Sample Vs. Along Track Sample: tvt
    cdf1 = coast.CDF(alt.dataset.interp_ssh.values[0:50])
    cdf2 = coast.CDF(alt.dataset.sla_unfiltered.values[0:50])