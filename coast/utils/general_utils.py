from dask import delayed
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
from warnings import warn
import copy
import scipy as sp
from .logging_util import get_slug, debug, info, warn, error
import sklearn.neighbors as nb

def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    '''
    # Estimation of geographical distance using the Haversine function.
    # Input can be single values or 1D arrays of locations. This
    # does NOT create a distance matrix but outputs another 1D array.
    # This works for either location vectors of equal length OR a single loc
    # and an arbitrary length location vector.
    #
    # lon1, lat1 :: Location(s) 1.
    # lon2, lat2 :: Location(s) 2.
    '''

    # Convert to radians for calculations
    lon1 = xr.ufuncs.deg2rad(lon1)
    lat1 = xr.ufuncs.deg2rad(lat1)
    lon2 = xr.ufuncs.deg2rad(lon2)
    lat2 = xr.ufuncs.deg2rad(lat2)

    # Latitude and longitude differences
    dlat = (lat2 - lat1) / 2
    dlon = (lon2 - lon1) / 2

    # Haversine function.
    distance = xr.ufuncs.sin(dlat) ** 2 + xr.ufuncs.cos(lat1) * \
               xr.ufuncs.cos(lat2) * xr.ufuncs.sin(dlon) ** 2
    distance = 2 * 6371.007176 * xr.ufuncs.arcsin(xr.ufuncs.sqrt(distance))

    return distance

def remove_indices_by_mask(A, mask):
    '''
    Removes indices from a 2-dimensional array, A, based on true elements of
    mask. A and mask variable should have the same shape.
    '''
    A = np.array(A).flatten()
    mask = np.array(mask, dtype=bool).flatten()
    array_removed = A[~mask]
        
    return array_removed

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

def dataarray_time_slice(data_array, date0, date1):
    ''' Takes an xr.DataArray object and returns a new object with times 
    sliced between dates date0 and date1. date0 and date1 may be a string or
    datetime type object.'''
    if date0 is None and date1 is None:
        return data_array
    else: 
        data_array_sliced = data_array.swap_dims({'t_dim':'time'})
        time_max = data_array.time.max().values
        time_min = data_array.time.min().values
        if date0 is None:
            date0 = time_min
        if date1 is None:
            date1 = time_max
        data_array_sliced = data_array_sliced.sel(time = slice(date0, date1))
        data_array_sliced = data_array_sliced.swap_dims({'time':'t_dim'})
        return data_array_sliced