'''
Python definitions used to aid in the calculation of Continuous Ranked
Probability Score.

*Methods Overview*
    -> crps_sonf_fixed(): Single obs neighbourhood forecast CRPS for fixed obs
    -> crps_song_moving(): Same as above for moving obs
'''

import numpy as np
import xarray as xr
from . import general_utils

def crps_empirical(sample, obs):
        """Calculates CRPS for a single observations against a sample of values.
        This sample of values may be an ensemble of model forecasts or a model
        neighbourhood. This is a comparison of a Heaviside function defined by
        the observation value and an Empirical Distribution Function (EDF)
        defined by the sample of values. This sample is sorted to create the 
        EDF.
        
        The calculation method is that outlined by Hersbach et al. (2000). 
        Each member of a supplied sample is weighted equally.
        
        Args:
            sample (array): Array of points (ensemble or neighbourhood)
            xa (float): A single 'observation' value which to compare against
                        CDF.

        Returns:
            A single CRPS value.
        """

        def calc(alpha, beta, p):
            return alpha * p**2 + beta*(1 - p)**2
        
        xa = float(obs)
        crps_integral = 0
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        sample_size = len(sample)
        
        alpha = np.zeros(sample_size-1)
        beta= np.zeros(sample_size-1)
        # sample[1:] = upper bounds, and vice versa
        
        tmp = sample[1:] - sample[:-1]
        tmp_logic = sample[1:]<xa
        alpha[tmp_logic] = tmp[tmp_logic]
                               
        tmp_logic = sample[:-1]>xa
        beta[tmp_logic] = tmp[tmp_logic]
        
        tmp_logic = ( sample[1:]>xa )*( sample[:-1]<xa )
        tmp = xa - sample[:-1]
        alpha[tmp_logic] = tmp[tmp_logic]
        tmp = sample[1:] - xa
        beta[tmp_logic] = tmp[tmp_logic]
        
        p = np.arange(1,sample_size)/sample_size
        c = alpha*p**2 + beta*(1-p)**2
        crps_integral = np.sum(c)
        
        # Intervals 0 and N, where p = 0 and 1 respectively
        if xa < sample[0]:
            p=0
            alpha = 0
            beta = sample[0] - xa
            crps = calc(alpha, beta, p)
            crps_integral += crps
        elif xa > sample[-1]:
            p=1
            alpha = xa - sample[-1]
            beta = 0
            crps = calc(alpha, beta, p)
            crps_integral += crps
            
        crps = crps_integral
            
        return crps
    
def crps_empirical_loop(sample, obs):
        """Like crps_empirical, however a loop is used instead of numpy 
        boolean indexing. For large samples, will be slower but consume less
        memory.
        """
        
        def calc(alpha, beta, p):
            return alpha * p**2 + beta*(1 - p)**2
        crps_integral = 0
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        sample_size = len(sample)
        
        # All intervals within range of the sample distribution
        for ii in range(0, sample_size-1):
            p = (ii+1)/sample_size
            if obs > sample[ii+1]:
                alpha = sample[ii+1] - sample[ii]
                beta = 0
            elif obs < sample[ii]:
                alpha = 0
                beta = sample[ii+1] - sample[ii]
            else:
                alpha = obs - sample[ii]
                beta = sample[ii+1] - obs
            crps = calc(alpha, beta, p)
            crps_integral += crps
        # Intervals 0 and N, where p = 0 and 1 respectively
        if obs < sample[0]:
            p=0
            alpha = 0
            beta = sample[0] - obs
            crps = calc(alpha, beta, p)
            crps_integral += crps
        elif obs > sample[-1]:
            p=1
            alpha = obs - sample[-1]
            beta = 0
            crps = calc(alpha, beta, p)
            crps_integral += crps
            
        return crps_integral
    
def crps_sonf_fixed( mod_array, obs_lon, obs_lat, obs_var, obs_time, 
                      nh_radius: float, cdf_type:str, time_interp:str
    ):
    '''
    Handles the calculation of single-observation neighbourhood forecast CRPS
    for a time series at a fixed observation location. Differs from 
    crps_sonf_moving in that it only need calculate a model neighbourhood once.

    Parameters
    ----------
    mod_array   : (xarray DataArray) DataArray from a Model Dataset
    obs_lon     : (float) Longitude of fixed observation point
    obs_lat     : (float) Latitude of fixed observation point
    obs_var     : (array) of floatArray of variable values, e.g time series
    obs_time    : (array) of datetimeArray of times, corresponding to obs_var
    nh_radius   : (float) Neighbourhood radius in km
    cdf_type    : (str) Type of CDF to use for model data. Either 'empirical' 
                   or 'theoretical'.
    time_interp : (str) Type of time interpolation to use

    Returns
    -------
    crps_list     : Array of CRPS values
    n_model_pts   : Array containing the number of model points used for 
                    each CRPS value
    contains_land : Array of bools indicating where a model neighbourhood 
                    contained land.

    '''

    # Define output arrays
    n_neighbourhoods = obs_var.shape[0] 
    crps_list     = np.zeros( n_neighbourhoods )*np.nan
    n_model_pts   = np.zeros( n_neighbourhoods )*np.nan
    contains_land = np.zeros( n_neighbourhoods , dtype=bool)

    # Loop over neighbourhoods
    neighbourhood_indices = np.arange(0,n_neighbourhoods)
    
    # Get model neighbourhood subset using specified method
    subset_ind = general_utils.subset_indices_by_distance(
                     mod_array.longitude.values, mod_array.latitude.values, 
                     obs_lon, obs_lat, nh_radius)
    mod_subset = mod_array.isel(y_dim = subset_ind[0],
                                  x_dim = subset_ind[1])
    mod_subset = mod_subset.swap_dims({'t_dim':'time'})
    
    # Check that the model neighbourhood contains points
    if subset_ind[0].shape[0] == 0 or subset_ind[1].shape[0] == 0:
        crps_list = crps_list*np.nan
    else:
        # Subset model data in time and space: model -> obs
        for ii in neighbourhood_indices:
            
                mod_subset_time = mod_subset.interp(
                                  time = obs_time[ii], method = time_interp,
                                  kwargs={'fill_value':'extrapolate'})
                
                #Check if neighbourhood contains a land value (TODO:mask)
                if any(np.isnan(mod_subset_time)):
                    contains_land[ii] = True
                # Check that neighbourhood contains a value
                if all(np.isnan(mod_subset_time)):
                    pass
                    # Calculate CRPS and put into output array
                    crps_list[ii] = crps_empirical(mod_subset_time, 
                                                   obs_var[ii])
                    n_model_pts[ii] = int(mod_subset.shape[0])

    return crps_list, n_model_pts, contains_land

def crps_sonf_moving( mod_array, obs_lon, obs_lat, obs_var, obs_time, 
                      nh_radius: float, cdf_type:str, time_interp:str
    ):
    '''
    Handles the calculation of single-observation neighbourhood forecast CRPS
    for a moving observation instrument. Differs from crps_sonf_fixed in that 
    latitude and longitude are arrays of locations.

    Parameters
    ----------
    mod_array   : (xarray DataArray) DataArray from a Model Dataset
    obs_lon     : (array) Longitudes of fixed observation point
    obs_lat     : (array) Latitudes of fixed observation point
    obs_var     : (array) of floatArray of variable values, e.g time series
    obs_time    : (array) of datetimeArray of times, corresponding to obs_var
    nh_radius   : (float) Neighbourhood radius in km
    cdf_type    : (str) Type of CDF to use for model data. Either 'empirical' 
                   or 'theoretical'.
    time_interp : (str) Type of time interpolation to use

    Returns
    -------
    crps_list     : Array of CRPS values
    n_model_pts   : Array containing the number of model points used for 
                    each CRPS value
    contains_land : Array of bools indicating where a model neighbourhood 
                    contained land.
    '''

    # Define output arrays
    n_neighbourhoods = obs_var.shape[0] 
    crps_list     = np.zeros( n_neighbourhoods )*np.nan
    n_model_pts   = np.zeros( n_neighbourhoods )*np.nan
    contains_land = np.zeros( n_neighbourhoods , dtype=bool)

    # Loop over neighbourhoods
    neighbourhood_indices = np.arange(0,n_neighbourhoods)
    for ii in neighbourhood_indices:
        
        # Neighbourhood centre
        cntr_lon = obs_lon[ii]
        cntr_lat = obs_lat[ii]
    
        # Get model neighbourhood subset using specified method
        subset_ind = general_utils.subset_indices_by_distance(
                         mod_array.longitude.values, mod_array.latitude.values, 
                         cntr_lon, cntr_lat, nh_radius)
        
        # Check that the model neighbourhood contains points
        if subset_ind[0].shape[0] == 0 or subset_ind[1].shape[0] == 0:
            crps_list[ii] = np.nan
        else:
            # Subset model data in time and space: model -> obs
            mod_subset = mod_array.isel(y_dim = subset_ind[0],
                                        x_dim = subset_ind[1])
            mod_subset = mod_subset.swap_dims({'t_dim':'time'})
            mod_subset = mod_subset.interp(
                             time = obs_time[ii], method = time_interp,
                             kwargs={'fill_value':'extrapolate'})
            
            #Check if neighbourhood contains a land value (TODO:mask)
            if any(np.isnan(mod_subset)):
                contains_land[ii] = True
            # Check that neighbourhood contains a value
            if all(np.isnan(mod_subset)):
                pass
            else:
                # Calculate CRPS and put into output array
                crps_list[ii] = crps_empirical(mod_subset, obs_var[ii])
                n_model_pts[ii] = int(mod_subset.shape[0])
                
    return crps_list, n_model_pts, contains_land

def subset_indices_by_distance(
        longitude, latitude, centre_lon: float, centre_lat: float, 
        radius: float
    ):
    """
    This method returns a `tuple` of indices within the `radius` of the lon/lat point given by the user.

    Distance is calculated as haversine - see `self.calculate_haversine_distance`

    :param centre_lon: The longitude of the users central point
    :param centre_lat: The latitude of the users central point
    :param radius: The haversine distance (in km) from the central point
    :return: All indices in a `tuple` with the haversine distance of the central point
    """

    # Calculate the distances between every model point and the specified
    # centre. Calls another routine dist_haversine.

    dist = calculate_haversine_distance(centre_lon, centre_lat, 
                                        longitude, latitude)
    indices_bool = dist < radius
    indices = np.where(indices_bool.compute())

    return xr.DataArray(indices[0]), xr.DataArray(indices[1])

def calculate_haversine_distance(lon1, lat1, lon2, lat2, r = 6371.00717):
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
    distance = xr.ufuncs.sin(dlat) ** 2 + xr.ufuncs.cos(lat1) *  \
        xr.ufuncs.cos(lat2) * xr.ufuncs.sin(dlon) ** 2
    distance = 2 * r * xr.ufuncs.arcsin(xr.ufuncs.sqrt(distance))

    return distance
