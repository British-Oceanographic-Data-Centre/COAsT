from dask import delayed
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
from warnings import warn
import copy
import scipy as sp
import sklearn.neighbors as nb

class UTILS():

    @staticmethod
    def remove_indices_by_mask(A, mask):
        '''
        Removes indices from a 2-dimensional array, A, based on true elements of
        mask. A and mask variable should have the same shape.
        '''
        A = np.array(A).flatten()
        mask = np.array(mask, dtype=bool).flatten()
        array_removed = A[~mask]
            
        return array_removed
        
    @staticmethod
    def reinstate_indices_by_mask(array_removed, mask, fill_value=np.nan):
        '''
        Rebuilds a 2D array from a 1D array created using remove_indices_by_mask().
        False elements of mask will be populated using array_removed. MAsked
        indices will be replaced with fill_value
        '''
        array_removed = np.array(array_removed)
        original_shape = mask.shape
        mask = np.array(mask, dtype=bool).flatten()
        A = np.zeros(mask.shape)
        A[~mask] = array_removed
        A[mask] = fill_value
        A = A.reshape(original_shape)
        return A
        
    @staticmethod
    def nearest_xy_indices(mod_lon, mod_lat, new_lon, new_lat, 
                           mask = None):
        '''
        Obtains the x and y indices of the nearest model points to specified
        lists of longitudes and latitudes. Makes use of sklearn.neighbours
        and its BallTree haversine method. 
        
        Example Useage
        ----------
        # Get indices of model points closest to altimetry points
        ind_x, ind_y = nemo.nearest_indices(altimetry.dataset.longitude,
                                            altimetry.dataset.latitude)
        # Nearest neighbour interpolation of model dataset to these points
        interpolated = nemo.dataset.isel(x_dim = ind_x, y_dim = ind_y)
        
        Parameters
        ----------
        mod_lon (2D array): Model longitude (degrees) array (2-dimensional)
        mod_lat (2D array): Model latitude (degrees) array (2-dimensions)
        new_lon (1D array): Array of longitudes (degrees) to compare with model
        new_lat (1D array): Array of latitudes (degrees) to compare with model
        mask (2D array): Mask array. Where True (or 1), elements of mod_lons
                         and mod lats will be removed prior to finding nearest
                         neighbours. e.g. if the nearest ocean point is 
            
        Returns
        -------
        Array of x indices, Array of y indices
        '''
        # Cast lat/lon to numpy arrays in case xarray things
        new_lon = np.array(new_lon)
        new_lat = np.array(new_lat)
        mod_lon = np.array(mod_lon)
        mod_lat = np.array(mod_lat)
        original_shape = mod_lon.shape
        
        # If a mask is supplied, remove indices from arrays.
        if mask is None:
            mod_lon = mod_lon.flatten()
            mod_lat = mod_lat.flatten()
        else:
            mod_lon[mask] = np.nan
            mod_lat[mask] = np.nan
            mod_lon = mod_lon.flatten()
            mod_lat = mod_lat.flatten()
        
        # Put lons and lats into 2D location arrays for BallTree: [lat, lon]
        mod_loc = np.vstack((mod_lat, mod_lon)).transpose()
        new_loc = np.vstack((new_lat, new_lon)).transpose()
        
        # Convert lat/lon to radians for BallTree
        mod_loc = np.radians(mod_loc)
        new_loc = np.radians(new_loc)
        
        # Do nearest neighbour interpolation using BallTree (gets indices)
        tree = nb.BallTree(mod_loc, leaf_size=5, metric='haversine')
        _, ind_1d = tree.query(new_loc, k=1)
        
        # Get 2D indices from 1D index output from BallTree
        ind_y, ind_x = np.unravel_index(ind_1d, original_shape)
        ind_x = xr.DataArray(ind_x.squeeze())
        ind_y = xr.DataArray(ind_y.squeeze())
        return ind_x, ind_y
    
