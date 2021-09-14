"""
WCSSP_India_example_plot.py

India subcontinent maritime domain.
WCSSP India configuration

Simple plot of sea surface temperature
"""

#%%
import coast
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # colormap fiddling

#################################################
#%%  Loading  data
#################################################

dir_nam = "/projectsa/COAsT/NEMO_example_data/MO_INDIA/"
fil_nam = "ind_1d_cat_20180101_20180105_25hourm_grid_T.nc"
dom_nam = "domain_cfg_wcssp.nc"


sci_t = coast.NEMO(
    dir_nam + fil_nam, dir_nam + dom_nam, grid_ref="t-grid", multiple=False
)

#%% Plot
fig = plt.figure()

plt.pcolormesh(
    sci_t.dataset.longitude,
    sci_t.dataset.latitude,
    sci_t.dataset.temperature.isel(t_dim=0).isel(z_dim=0),
)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("WCSSP India SST")
plt.colorbar()
plt.show()
fig.savefig("WCSSP_India_SST.png", dpi=120)
