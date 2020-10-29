import coast
import coast.general_utils as gu
import numpy as np
import matplotlib.pyplot as plt

fn_detided = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/detided.nc'
fn_domain = '/Users/Dave/Documents/Projects/WCSSP/Data/domain_cfg_wcssp.nc'
fn_alt = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*j3*'

nemo = coast.NEMO(fn_detided , fn_domain, grid_ref = 't-grid')
nemo = nemo.isel(x_dim = np.arange(0,1760,2), y_dim = np.arange(0,1100,2))

alt = coast.ALTIMETRY(fn_alt, multiple=True)
ind = alt.subset_indices_lonlat_box((65,99),(3.5,27))
alt.dataset['no_dac'] = alt.dataset.sla_unfiltered + alt.dataset.dac
alt = alt.isel(t_dim=ind[::5])

alon = alt.dataset.longitude
alat = alt.dataset.latitude
lon0 = alt.dataset.longitude[300]
lat0 = alt.dataset.latitude[300]
time0 = alt.dataset.time[300]
ssh = nemo.dataset.ssh
mlon = ssh.longitude
mlat = ssh.latitude
radius=200

ssh.load()

# Model Sample
x,y = gu.subset_indices_by_distance(mlon, mlat, lon0, lat0, radius)
msub = ssh.isel(x_dim=x, y_dim=y)
msub = gu.interpolate_in_time(msub, time0)

# Observation Sample
t = gu.subset_indices_by_distance(alon, alat, lon0, lat0, radius)
asub = alt.isel(t_dim=t)

# Obs operator
alt.obs_operator(ssh, time_interp='linear')

#cdfs
mcdf = coast.CDF(msub.values)
acdf = coast.CDF(asub.dataset.sla_unfiltered.values)


for radii in np.arange(0,np.arange())
cdf1 = coast.CDF(alt.dataset.interp_ssh.values[0:50])
cdf2 = coast.CDF(alt.dataset.sla_unfiltered.values[0:50])