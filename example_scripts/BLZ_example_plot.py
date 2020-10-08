"""
BLZ_example_plot.py 

Make simple Belize SSH plot.

"""

#%%
import coast
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.colors as colors # colormap fiddling

#################################################
#%%  Loading  data
#################################################


dir_nam = "/projectsa/accord/GCOMS1k/OUTPUTS/BLZE12_02/2015/"
fil_nam = "BLZE12_1h_20151101_20151130_grid_T.nc"
dom_nam = "/projectsa/accord/GCOMS1k/INPUTS/BLZE12_C1/domain_cfg.nc"
        

sci_t = coast.NEMO(dir_nam + fil_nam, \
        dom_nam, grid_ref='t-grid', multiple=False)

sci_u = coast.NEMO(dir_nam + fil_nam.replace('grid_T','grid_U'), \
        dom_nam, grid_ref='u-grid', multiple=False)

sci_v = coast.NEMO(dir_nam + fil_nam.replace('grid_T','grid_V'), \
        dom_nam, grid_ref='v-grid', multiple=False)

# create an empty w-grid object, to store stratification
sci_w = coast.NEMO( fn_domain = dom_nam, grid_ref='w-grid')



#%% Plot
plt.pcolormesh( sci_t.dataset.ssh.isel(t_dim=0)) ;plt.show()

