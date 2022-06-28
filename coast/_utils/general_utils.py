"""A general utility file."""
import xarray as xr
import numpy as np
import sklearn.neighbors as nb
import pandas as pd


def determine_season(t):
    """
    Determine season (or array of seasons) from a time (Datetime or xarray)
    object. Put in an array of times, get out an array of seasons.
    """
    season_dict = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    pd_month = pd.to_datetime(t).month
    pd_season = [season_dict[pp] for pp in pd_month]
    return np.array(pd_season)


def subset_indices_by_distance_balltree(longitude, latitude, centre_lon, centre_lat, radius: float, mask=None):
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
    """
    # change inputs to numpy
    longitude = np.array(longitude)
    latitude = np.array(latitude)
    centre_lon = np.array(centre_lon)
    centre_lat = np.array(centre_lat)
    # Calculate radius in radians
    earth_radius = 6371
    r_rad = radius / earth_radius
    # For reshaping indices at the end
    original_shape = longitude.shape
    # Check if radius centres are numpy arrays. If not, make them into ndarrays
    if not isinstance(centre_lon, np.ndarray):
        centre_lat = np.array(centre_lat)
        centre_lon = np.array(centre_lon)
    # Determine number of centres provided
    n_pts = 1 if centre_lat.shape == () else len(centre_lat)
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
    if n_pts == 1:
        centre = np.array([[centre_lat, centre_lon]])
    else:
        centre = np.vstack((centre_lat, centre_lon)).transpose()
    centre = np.radians(centre)
    # Do nearest neighbour interpolation using BallTree (gets indices)
    tree = nb.BallTree(locs, leaf_size=2, metric="haversine")
    ind_1d = tree.query_radius(centre, r=r_rad)
    if len(original_shape) == 1:
        return ind_1d
    else:
        # Get 2D indices from 1D index output from BallTree
        ind_y = []
        ind_x = []
        for ii in np.arange(0, n_pts):
            x_tmp, y_tmp = np.unravel_index(ind_1d[ii], original_shape)
            ind_x.append(x_tmp.squeeze())
            ind_y.append(y_tmp.squeeze())
        if n_pts == 1:
            return ind_x[0], ind_y[0]
        else:
            return ind_x, ind_y


def subset_indices_by_distance(longitude, latitude, centre_lon: float, centre_lat: float, radius: float):
    """
    This method returns a `tuple` of indices within the `radius` of the
    lon/lat point given by the user.
    Scikit-learn BallTree is used to obtain indices.
    :param longitude: The longitude of the users central point
    :param latitude: The latitude of the users central point
    :param radius: The haversine distance (in km) from the central point
    :return: All indices in a `tuple` with the haversine distance of the
            central point
    """

    # Calculate the distances between every model point and the specified
    # centre. Calls another routine dist_haversine.
    dist = calculate_haversine_distance(centre_lon, centre_lat, longitude, latitude)
    indices_bool = dist < radius
    indices = np.where(indices_bool)

    if len(longitude.shape) == 1:
        return xr.DataArray(indices[0])
    else:
        return xr.DataArray(indices[0]), xr.DataArray(indices[1])


def compare_angles(a1, a2, degrees=True):
    """
    # Compares the difference between two angles. e.g. it is 2 degrees between
    # 359 and 1 degree. If degrees = False then will treat angles as radians.
    """

    if not degrees:
        a1 = np.degrees(a1)
        a2 = np.degrees(a2)

    diff = 180 - np.abs(np.abs(a1 - a2) - 180)

    return diff


def cartesian_to_polar(x, y, degrees=True):
    """
    # Conversion of cartesian to polar coordinate system
    # Output theta is in radians
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if degrees:
        theta = np.rad2deg(theta)
    return r, theta


def polar_to_cartesian(r, theta, degrees=True):
    """
    # Conversion of polar to cartesian coordinate system
    # Input theta must be in radians
    """
    if degrees:
        theta = np.deg2rad(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def subset_indices_lonlat_box(array_lon, array_lat, lon_min, lon_max, lat_min, lat_max):
    ind_lon = np.logical_and(array_lon <= lon_max, array_lon >= lon_min)
    ind_lat = np.logical_and(array_lat <= lat_max, array_lat >= lat_min)
    ind = np.where(np.logical_and(ind_lon, ind_lat))
    return ind


def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    """
    # Estimation of geographical distance using the Haversine function.
    # Input can be single values or 1D arrays of locations. This
    # does NOT create a distance matrix but outputs another 1D array.
    # This works for either location vectors of equal length OR a single loc
    # and an arbitrary length location vector.
    #
    # lon1, lat1 :: Location(s) 1.
    # lon2, lat2 :: Location(s) 2.
    """

    # Convert to radians for calculations
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # Latitude and longitude differences
    dlat = (lat2 - lat1) / 2
    dlon = (lon2 - lon1) / 2

    # Haversine function.
    distance = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    distance = 2 * 6371.007176 * np.arcsin(np.sqrt(distance))

    return distance


def remove_indices_by_mask(array, mask):
    """
    Removes indices from a 2-dimensional array, A, based on true elements of
    mask. A and mask variable should have the same shape.
    """
    array = np.array(array).flatten()
    mask = np.array(mask, dtype=bool).flatten()
    array_removed = array[~mask]

    return array_removed


def reinstate_indices_by_mask(array_removed, mask, fill_value=np.nan):
    """
    Rebuilds a 2D array from a 1D array created using remove_indices_by_mask().
    False elements of mask will be populated using array_removed. MAsked
    indices will be replaced with fill_value
    """
    array_removed = np.array(array_removed)
    original_shape = mask.shape
    mask = np.array(mask, dtype=bool).flatten()
    array = np.zeros(mask.shape)
    array[~mask] = array_removed
    array[mask] = fill_value
    array = array.reshape(original_shape)
    return array


def nearest_indices_2d(mod_lon, mod_lat, new_lon, new_lat, mask=None):
    """
    Obtains the 2 dimensional indices of the nearest model points to specified
    lists of longitudes and latitudes. Makes use of sklearn.neighbours
    and its BallTree haversine method. Ensure there are no NaNs in
    input longitude/latitude arrays (or mask them using "mask"")

    Example Usage
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
    """
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
        mod_lon = remove_indices_by_mask(mod_lon, mask)
        mod_lat = remove_indices_by_mask(mod_lat, mask)
        # If we are masking, we want to preserve the original indices so that
        # we can get them back at the end (since masked points are removed).
        cc, rr = np.meshgrid(np.arange(0, original_shape[1]), np.arange(0, original_shape[0]))
        cc = remove_indices_by_mask(cc, mask)
        rr = remove_indices_by_mask(rr, mask)

    # Put lons and lats into 2D location arrays for BallTree: [lat, lon]
    mod_loc = np.vstack((mod_lat, mod_lon)).transpose()
    new_loc = np.vstack((new_lat, new_lon)).transpose()

    # Convert lat/lon to radians for BallTree
    mod_loc = np.radians(mod_loc)
    new_loc = np.radians(new_loc)

    # Do nearest neighbour interpolation using BallTree (gets indices)
    tree = nb.BallTree(mod_loc, leaf_size=5, metric="haversine")
    _, ind_1d = tree.query(new_loc, k=1)

    if mask is None:
        # Get 2D indices from 1D index output from BallTree
        ind_y, ind_x = np.unravel_index(ind_1d, original_shape)
    else:
        ind_y = rr[ind_1d]
        ind_x = cc[ind_1d]

    ind_x = xr.DataArray(ind_x.squeeze())
    ind_y = xr.DataArray(ind_y.squeeze())

    return ind_x, ind_y


def data_array_time_slice(data_array, date0, date1):
    """Takes an xr.DataArray object and returns a new object with times
    sliced between dates date0 and date1. date0 and date1 may be a string or
    datetime type object."""
    if date0 is None and date1 is None:
        return data_array
    else:
        data_array_sliced = data_array.swap_dims({"t_dim": "time"})
        time_max = data_array.time.max().values
        time_min = data_array.time.min().values
        if date0 is None:
            date0 = time_min
        if date1 is None:
            date1 = time_max
        data_array_sliced = data_array_sliced.sel(time=slice(date0, date1))
        data_array_sliced = data_array_sliced.swap_dims({"time": "t_dim"})
        return data_array_sliced


def day_of_week(date: np.datetime64 = None):
    """Return the day of the week (3 letter str)"""
    if date is None:
        date = np.datetime64("now")

    val = (np.datetime64(date, "D") - np.datetime64(date, "W")).astype(int)
    if val == 0:
        return "Thu"
    elif val == 1:
        return "Fri"
    elif val == 2:
        return "Sat"
    elif val == 3:
        return "Sun"
    elif val == 4:
        return "Mon"
    elif val == 5:
        return "Tue"
    elif val == 6:
        return "Wed"
