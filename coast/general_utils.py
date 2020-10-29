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

def interpolate_in_space(model_array, new_lon, new_lat, mask=None):
    '''
    Interpolates a provided xarray.DataArray in space to new longitudes
    and latitudes using a nearest neighbour method (BallTree). This is for
    interpolating to a 1D sequence of point locations and should not be used
    for regridding. Model_array is assumed to contain coordinates called
    'longitude' and 'latitude' and corresponding dimensions x_dim and y_dim
    (as is the COAsT data format).
    
    Example Usage
    ----------
    # Get an interpolated DataArray for temperature onto two locations
    interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                             [0,1], [45,46])
    Parameters
    ----------
    model_array (xr.DataArray): Model variable DataArray to interpolate
    new_lons (1Darray): Array of longitudes (degrees) to compare with model
    new_lats (1Darray): Array of latitudes (degrees) to compare with model
    mask (2D array): Mask array. Where True (or 1), elements of array will
                 not be included. For example, use to mask out land in 
                 case it ends up as the nearest point.
    
    Returns
    -------
    Interpolated DataArray
    '''
    debug(f"Interpolating {get_slug(model_array)} in space with nearest neighbour")
    # Get nearest indices
    ind_x, ind_y = nearest_indices_2D(model_array.longitude,
                                      model_array.latitude,
                                      new_lon, new_lat, mask=mask)
    
    # Geographical interpolation (using BallTree indices)
    ind_x = xr.DataArray(ind_x)
    ind_y = xr.DataArray(ind_y)
    interpolated = model_array.isel(x_dim=ind_x, y_dim=ind_y)
    if 'dim_0' in interpolated.dims:
        interpolated = interpolated.rename({'dim_0':'interp_dim'})
    return interpolated

def interpolate_in_time(model_array, new_times, 
                           interp_method = 'nearest', extrapolate=True):
    '''
    Interpolates a provided xarray.DataArray in time to new python
    datetimes using a specified scipy.interpolate method. Model_array is
    assumed to contain a t_dim dimension and a time coordinate.
    
    Example Useage
    ----------
    # Get an interpolated DataArray for temperature onto altimetry times
    new_times = altimetry.dataset.time
    interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                             new_times)
    Parameters
    ----------
    model_array (xr.DataArray): Model variable DataArray to interpolate
    new_times (array): New times to interpolate to (array of datetimes)
    interp_method (str): Interpolation method
    
    Returns
    -------
    Interpolated DataArray
    '''
    debug(f"Interpolating {get_slug(model_array)} in time with method \"{interp_method}\"")
    # Time interpolation
    interpolated = model_array.swap_dims({'t_dim':'time'})
    if extrapolate:
        interpolated = interpolated.interp(time = new_times,
                                           method = interp_method,
                                           kwargs={'fill_value':'extrapolate'})
    else:
        interpolated = interpolated.interp(time = new_times,
                                       method = interp_method)
    # interpolated = interpolated.swap_dims({'time':'t_dim'})  # TODO Do something with this or delete it
    
    return interpolated

def subset_indices_by_distance_BT(longitude, latitude, centre_lon, centre_lat, 
        radius: float, mask=None
    ):
    """
    Returns the indices of points that lie within a specified radius (km) of
    central latitude and longitudes. This makes use of BallTree.query_radius.
    
    Parameters
    ----------
    longitude   : (numpy.ndarray) longitudes in degrees
    latitude    : (numpy.ndarray) latitudes in degrees
    centre_lon  : Central longitude. Can be single value or array of values
    centre_lat  : Central latitude. Can be single value or array of values
    radius      : (float) Radius in km within which to find indices
    mask        : (numpy.ndarray) of same dimension as longitude and latitude.
                  If specified, will mask out points from the routine.
    Returns
    -------
        Returns an array of indices corresponding to points within radius.
        If more than one central location is specified, this will be a list
        of index arrays. Each element of which corresponds to one centre.
    If longitude is 1D:
        Returns one array of indices per central location
    If longitude is 2D:
        Returns arrays of x and y indices per central location.
        ind_y corresponds to row indices of the original input arrays.
        
    DB: This is really slow at the moment. But the bottleneck seems to be right
    at the start somewhere.. Best stick to subset_indices_by_distance for now.
    """
    # Calculate radius in radians
    earth_radius = 6371
    r_rad = radius/earth_radius
    # For reshaping indices at the end
    original_shape = longitude.shape
    # Check if radius centres are numpy arrays. If not, make them into ndarrays
    if not isinstance(centre_lon, (np.ndarray)):
        centre_lat = np.array(centre_lat)
        centre_lon = np.array(centre_lon)
    # Determine number of centres provided
    n_pts = 1 if centre_lat.shape==() else len(centre_lat)
    # If a mask is supplied, remove indices from arrays. Flatten input ready
    # for BallTree
    if mask is None:
        longitude = longitude.flatten()
        latitude = latitude.flatten()
    else:
        longitude[mask] = np.nan
        latitude[mask] = np.nan
        longitude = longitude.flatten()
        latitude = latitude.flatten()
    # Put lons and lats into 2D location arrays for BallTree: [lat, lon]
    locs = np.vstack((latitude, longitude)).transpose()
    locs = np.radians(locs)
    # Construct central input to BallTree.query_radius
    if n_pts==1:
        centre = np.array([[centre_lat, centre_lon]])
    else:
        centre = np.vstack((centre_lat, centre_lon)).transpose()
    centre = np.radians(centre)
    # Do nearest neighbour interpolation using BallTree (gets indices)
    tree = nb.BallTree(locs, leaf_size=2, metric='haversine')
    ind_1d = tree.query_radius(centre, r = r_rad)
    if len(original_shape) == 1:
        return ind_1d[0]
    else:
        # Get 2D indices from 1D index output from BallTree
        ind_y = []
        ind_x = []
        for ii in np.arange(0,n_pts):
            x_tmp, y_tmp = np.unravel_index(ind_1d[ii], original_shape)
            ind_x.append(x_tmp.squeeze())
            ind_y.append(y_tmp.squeeze())
        if n_pts==1:
            return ind_x[0], ind_y[0]
        else:
            return ind_x, ind_y

def subset_indices_by_distance(
        longitude, latitude, centre_lon: float, centre_lat: float, 
        radius: float
    ):
    """
    This method returns a `tuple` of indices within the `radius` of the 
    lon/lat point given by the user.

    Scikit-learn BallTree is used to obtain indices.

    :param centre_lon: The longitude of the users central point
    :param centre_lat: The latitude of the users central point
    :param radius: The haversine distance (in km) from the central point
    :return: All indices in a `tuple` with the haversine distance of the 
            central point
    """

    # Calculate the distances between every model point and the specified
    # centre. Calls another routine dist_haversine.

    dist = calculate_haversine_distance(centre_lon, centre_lat, 
                                        longitude, latitude)
    indices_bool = dist < radius
    indices = np.where(indices_bool)

    if len(longitude.shape) == 1:
        return xr.DataArray(indices[0])
    else:
        return xr.DataArray(indices[0]), xr.DataArray(indices[1])

def subset_indices_time_window(time_array, time_min, time_max):
    ''' Gets indices for time_array where time values sit in specified window.

    Parameters
    ----------
    time_array (1Darray) : Array of datetime or np.datetime64
    time_min   (1Darray) : Lower bound(s) of time window.
    time_max   (1Darray) : Upper bound(s) of time window.
    
    time_min and time_max may be either single datetime/datetime64 objects or
    an array-like object containing multiple datetime objects. In this case,
    the indices returned are the union of indices for each time window.

    Returns
    -------
    1Darray of indices
    '''
    
    if type(time_min) not in [np.ndarray, list, xr.core.dataarray.DataArray]:
        time_min = [time_min]
        time_max = [time_max]
        n_windows=1
    if type(time_array) == xr.core.dataarray.DataArray:
        time_array = time_array.values
    if type(time_min) == xr.core.dataarray.DataArray:
        time_min = time_min.values
        time_max = time_max.values
        
    indices = np.array([], dtype='int')
        
    for ii in range(0,len(time_min)):
        bool_array = (time_array >= time_min[ii])*(time_array <= time_max[ii])
        ind_tmp = np.where(bool_array)[0]
        indices = np.union1d(indices, ind_tmp)
    
    return indices

def subset_indices_lonlat_box(longitude, latitude, lon_bounds, lat_bounds):
    """
    Get indices corresponding to locations within a specified longitude and
    latitude box. This converts everything to numpy arrays and may be slow for
    large dataset. 
    
    *Note: For more efficient calculation, use find_j_i instead,
           however, the output may not be as accurate.
    *Note: This routine will not work over the longitude 'seam' and may give
           unexpected output over the poles. (TODO)
    
    Parameters
    ----------
    longitude (array) : 1D or 2D array of longitudes.
    latitude  (array) : 1D or 2D array of latitudes.
    lon_bounds (tuple) : Tuple of longitude bounds to define box.
    lat_bounds (tuple) : Tuple of latitude bounds to defin box.
    
    Returns
    -------
    Tuple of indices corresponding to each axis (in numpy order) of input.
    """
    # Check longitude and latitude have the same type
    if type(longitude) != type(latitude):
        print('Ensure that type(longitude) == type(latitude)')
        return
    
    # Determine type of inputs and allocation variables appropriately
    if type(longitude) == xr.core.dataarray.DataArray:
        lon = longitude.copy().values
        lat = latitude.values
    elif type(longitude) == np.ndarray:
        lon = longitude.copy()
        lat = latitude
    elif type(longitude) == list:
        lon = np.array(longitude)
        lat = np.array(latitude)
        
    # Shift longitude values to be consistent.
    lon[lon>180] = lon[lon>180] - 360
    lon[lon<-180] = lon[lon<-180] + 360
    
    # Do some boolean logic to identify points.
    bool_array = ( lon > lon_bounds[0] ) * ( lon < lon_bounds[1] ) \
               * ( lat > lat_bounds[0] ) * ( lat < lat_bounds[1] )
    
    # Indices are where all the above booleans are satisfied
    indices = np.where( bool_array )
    if len(indices)==1:
        return indices[0]
    else:
        return(indices)

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

def nearest_indices_2D(mod_lon, mod_lat, new_lon, new_lat, 
                       mask = None):
    '''
    Obtains the 2 dimensional indices of the nearest model points to specified
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
    mask (2D array): Mask array. Where True (or 1), elements of array will
                     not be included. For example, use to mask out land in 
                     case it ends up as the nearest point.
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
    tree = nb.BallTree(mod_loc, leaf_size=2, metric='haversine')
    _, ind_1d = tree.query(new_loc, k=1)
    
    # Get 2D indices from 1D index output from BallTree
    ind_y, ind_x = np.unravel_index(ind_1d, original_shape)
    
    return ind_x.squeeze(), ind_y.squeeze()

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