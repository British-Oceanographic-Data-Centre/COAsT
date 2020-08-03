import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
from .CRPS import CRPS
import sklearn.neighbors as nb

class STATS():
    
    def __init__(self, model, observations, 
                 mod_var_name:str, obs_var_name:str,
                 time_interp = 'nearest'):
        
        # Get data arrays
        mod_var = model.dataset[mod_var_name]
        obs_var = observations.dataset[obs_var_name]
        self.dataset = observations.dataset[['longitude','latitude']]
        
        # Cast to numpy arrays
        mod_lon = np.array(mod_var.longitude).flatten()
        mod_lat = np.array(mod_var.latitude).flatten()
        obs_lon = np.array(obs_var.longitude).flatten()
        obs_lat = np.array(obs_var.latitude).flatten()
        
        # Put lons and lats into a 2D array for BallTree
        mod_locs = np.vstack((mod_lat, mod_lon)).transpose()
        obs_locs = np.vstack((obs_lat, obs_lon)).transpose()
        
        # Do nearest neighbour interpolation using BallTree
        tree = nb.BallTree(mod_locs, leaf_size=5, metric='haversine')
        _, ind_1d = tree.query(obs_locs, k=1)
        
        # Get 2D indices from 1D index output from BallTree
        ind_y, ind_x = np.unravel_index(ind_1d, mod_var.longitude.shape)
        ind_y = ind_y.squeeze()
        ind_x = ind_x.squeeze()
        ind_y = xr.DataArray(ind_y)
        ind_x = xr.DataArray(ind_x)
        self.ind_y = ind_y
        self.ind_x = ind_x
        self.mod_locs = mod_locs
        self.obs_locs = obs_locs
        
        # Geographical interpolation (using BallTree indices)
        mod_interp = mod_var.isel(x_dim=ind_x, y_dim=ind_y)
        
        # Depth interpolation -> for now just take 0 index
        if 'z_dim' in mod_var.dims:
            mod_interp = mod_interp.isel(z_dim=0)
        
        # Time interpolation
        #if 't_dim' in mod_var.dims:
        #    mod_interp = mod_interp.rename({'t_dim':'time'})
        #    mod_interp = mod_interp.interp(time = obs_var['time'],
        #                                   method = time_interp,
        #                                   kwargs={'fill_value':'extrapolate'})
        #    mod_interp = mod_interp.rename({'time':'t_dim'})
            
        #tmp_ind = xr.DataArray(np.arange(0,250))
        #self.mod_interp = mod_interp.isel(t_dim=tmp_ind, dim_0=tmp_ind)
        self.mod_interp = mod_interp.copy()
        
        return
    
    def mae(self):
        pass
        return 