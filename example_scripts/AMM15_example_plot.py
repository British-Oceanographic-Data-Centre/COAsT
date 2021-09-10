"""
AMM15_example_plot.py 

Make simple AMM15 SST plot.

"""

#%%
import coast
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors # colormap fiddling

#################################################
#%%  Loading  data
#################################################


config = 'AMM15'
dir_nam = "/projectsa/NEMO/gmaya/2013p2/"
fil_nam = "20130415_25hourm_grid_T.nc"
dom_nam = "/projectsa/NEMO/gmaya/AMM15_GRID/amm15.mesh_mask.cs3x.nc"
config = "/work/jelt/GitHub/COAsT/example_files/example_t_nemo_config.json"
        
sci_t = coast.Gridded(dir_nam + fil_nam, 
        dom_nam, config=config  ) #, chunks=chunks)
chunks = {"x_dim":10, "y_dim":10, "t_dim":10}
sci_t.dataset.chunk(chunks)

# create an empty w-grid object, to store stratification
sci_w = coast.Gridded( fn_domain = dom_nam, config=config.replace("t_nemo","w_nemo")) #, chunks=chunks)
sci_w.dataset.chunk({"x_dim":10, "y_dim":10})


print('* Loaded ',config, ' data')

#################################################
#%% subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
print('* Extract North Sea subdomain')
ind_sci = sci_t.subset_indices([51,-4], [62,15])
sci_nwes_t = sci_t.isel(y_dim=ind_sci[0], x_dim=ind_sci[1]) #nwes = northwest europe shelf
ind_sci = sci_w.subset_indices([51,-4], [62,15])
sci_nwes_w = sci_w.isel(y_dim=ind_sci[0], x_dim=ind_sci[1]) #nwes = northwest europe shelf

#%% Apply masks to temperature and salinity
if config == 'AMM15':
    sci_nwes_t.dataset['temperature_m'] = sci_nwes_t.dataset.temperature.where( sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset['t_dim'].sizes) > 0) 
    sci_nwes_t.dataset['salinity_m'] = sci_nwes_t.dataset.salinity.where( sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset['t_dim'].sizes) > 0) 

else:
    # Apply fake masks to temperature and salinity
    sci_nwes_t.dataset['temperature_m'] = sci_nwes_t.dataset.temperature
    sci_nwes_t.dataset['salinity_m'] = sci_nwes_t.dataset.salinity



#%% Plots
fig = plt.figure()

plt.pcolormesh( sci_t.dataset.longitude, sci_t.dataset.latitude, sci_t.dataset.temperature.isel(z_dim=0).squeeze())
#plt.xlabel('longitude')
#plt.ylabel('latitude')
#plt.colorbar()
plt.axis('off')
plt.show()


fig.savefig('AMM15_SST_nocolorbar.png', dpi=120)


