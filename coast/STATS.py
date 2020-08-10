import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
from .CRPS import CRPS
import sklearn.neighbors as nb

class STATS():
    '''
    For basic comparisons between observed and modelled variable data. On
    initialisation, model data is interpolated from model coordinates to 
    model coordinates. This is done in both time and space (independently). 
    Spatial interpolation is done using a nearest neighbour interpolation
    (using sklearn.neighbours.BallTree's haversine method). Time interpolation
    is done using xarray.interp(), which uses scipy.interpolate.interp1d. The
    method used for time interpolation can be specified by the user, but by 
    default is nearest neighbour.
            
    Currently the object only holds interpolated model values in its own 
    xarray.dataset, which is of the same format as the inputted observation
    object. For depth, no interpolation is currently done, only the first
    index is taken.
    
    Example usage:
    --------------
    # Initialise COAsT objects
    nemo_t = coast.NEMO( fn_data, fn_domain, grid_ref='t-grid' )
    altimetry = coast.ALTIMETRY(fn_alt)
    
    # Create STATS object for sea surface height
    stats = coast.STATS(nemo_t, altimetry, 'sossheig', 'sla_filtered')
    
    # Interrogate object's dataset to access interpolated data.
    
    Parameters
    ----------
    model : model object (e.g. NEMO)
    observations : observations object (e.g. ALTIMETRY)
    mod_var: variable name string to use from model object
    obs_var: variable name string to use from observations object
    time_interp: time interpolation method (optional, default: 'nearest')
            This can take any string scipy.interpolate would take. e.g.
            'nearest', 'linear' or 'cubic'
    Returns
    -------
    Self (STATS object).
    '''  
    
    def __init__(self, model, observations, 
                 mod_var_name:str, obs_var_name:str,
                 time_interp = 'nearest'):
        
        # Get data arrays
        mod_var = model.dataset[mod_var_name]
        obs_var = observations.dataset[obs_var_name]
        self.dataset = observations.dataset[['longitude','latitude']]
        self.dataset.attrs = {}
        
        # Cast lat/lon to numpy arrays
        mod_lon = np.array(mod_var.longitude).flatten()
        mod_lat = np.array(mod_var.latitude).flatten()
        obs_lon = np.array(obs_var.longitude).flatten()
        obs_lat = np.array(obs_var.latitude).flatten()
        
        # Put lons and lats into 2D location arrays for BallTree: [lat, lon]
        mod_locs = np.vstack((mod_lat, mod_lon)).transpose()
        obs_locs = np.vstack((obs_lat, obs_lon)).transpose()
        
        # Convert lat/lon to radians for BallTree
        print('a')
        mod_locs = np.radians(mod_locs)
        obs_locs = np.radians(obs_locs)
        print('b')
        
        # Do nearest neighbour interpolation using BallTree (gets indices)
        tree = nb.BallTree(mod_locs, leaf_size=5, metric='haversine')
        _, ind_1d = tree.query(obs_locs, k=1)
        print('c')
        
        # Get 2D indices from 1D index output from BallTree
        ind_y, ind_x = np.unravel_index(ind_1d, mod_var.longitude.shape)
        ind_y = ind_y.squeeze()
        ind_x = ind_x.squeeze()
        ind_y = xr.DataArray(ind_y)
        ind_x = xr.DataArray(ind_x)
        print('d')
        
        # Geographical interpolation (using BallTree indices)
        mod_interp = mod_var.isel(x_dim=ind_x, y_dim=ind_y)
        print('e')
        
        # Depth interpolation -> for now just take 0 index
        if 'z_dim' in mod_var.dims:
            mod_interp = mod_interp.isel(z_dim=0)
        print('f')
        # Time interpolation
        if 't_dim' in mod_var.dims:
            mod_interp = mod_interp.rename({'t_dim':'time'})
            mod_interp = mod_interp.interp(time = obs_var['time'],
                                           method = time_interp,
                                           kwargs={'fill_value':'extrapolate'})
            mod_interp = mod_interp.rename({'time':'t_dim'})
        print('g')
            
        diag_len = mod_interp.shape[0]
        diag_ind = xr.DataArray(np.arange(0, diag_len))
        self.mod_interp = mod_interp.isel(dim_0=diag_ind, t_dim=diag_ind)

        self.dataset['interpolated'] = mod_interp
        return
