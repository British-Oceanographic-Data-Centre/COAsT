import coast
import numpy as np
import matplotlib.pyplot as plt

fn_detided = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/detided.nc'
fn_domain = '/Users/Dave/Documents/Projects/WCSSP/Data/domain_cfg_wcssp.nc'
fn_alt = '/Users/Dave/Documents/Projects/WCSSP/Data/fani/*j3*'

nemo = coast.NEMO(fn_detided , fn_domain, grid_ref = 't-grid')

alt = coast.ALTIMETRY(fn_alt, multiple=True)
ind = alt.subset_indices_lonlat_box((65,99),(3.5,27))
alt.dataset['no_dac'] = alt.dataset.sla_unfiltered + alt.dataset.dac
alt = alt.isel(t_dim=ind[::10])

radii = np.arange(5,20,2)
crps_list = []

for rr in radii:

    print(rr)
    crps = alt.crps(nemo, 'ssh', 'no_dac', 25)
    crps_list.append(crps)


crps_mean = []
# all_land = crps_list[-1].dataset.crps_contains_land.values

for aa in range(0,len(crps_list)):
     cc = crps_list[aa].dataset.crps
     #cc[all_land] = np.nan
     crps_mean.append(np.nanmean(cc))
    
crps_mean=np.array(crps_mean)