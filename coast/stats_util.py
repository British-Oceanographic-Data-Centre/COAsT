import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
from .CRPS import CRPS

def crps_sonf():
    return

def crps_sonf(
        mod_lon, mod_lat, mod_var, obs_lon, obs_lat, obs_var,
        nh_radius: float, nh_type: str, cdf_type:str, time_interp:str
):
    """
    """

    # Define output arrays
    n_neighbourhoods = obs_var.shape[0] 
    crps_list     = np.zeros( n_neighbourhoods )*np.nan
    n_model_pts   = np.zeros( n_neighbourhoods )*np.nan
    contains_land = np.zeros( n_neighbourhoods , dtype=bool)
    mod_cdf = None
    obs_cdf = None

    # Loop over neighbourhoods
    neighbourhood_indices = np.arange(0,n_neighbourhoods)
    for ii in neighbourhood_indices:
        
        print("\r Progress: [[ "+str(round(ii/n_neighbourhoods*100,2)) + 
              '% ]]', end=" ", flush=True)
        
        # Neighbourhood centre
        cntr_lon = obs_data.longitude[ii]
        cntr_lat = obs_data.latitude[ii]
    
        # Get model neighbourhood subset using specified method
        if nh_type == "radius":
            subset_ind = self.subset_indices_by_distance(model_data.longitude,
                              model_data.latitude, cntr_lon, cntr_lat, 
                              nh_radius)
        elif nh_type == "box":
            raise NotImplementedError
        
        # Check that the model neighbourhood contains points
        if subset_ind[0].shape[0] == 0 or subset_ind[1].shape[0] == 0:
            crps_list[ii] = np.nan
        else:
            # Subset model data in time and space: model -> obs
            mod_subset = model_data.isel(y_dim = subset_ind[0],
                                           x_dim = subset_ind[1])
            mod_subset = mod_subset.interp(time = obs_data['time'][ii],
                                               method = time_interp,
                                               kwargs={'fill_value':'extrapolate'})
            
            #Check if neighbourhood contains a land value (TODO:mask)
            if any(np.isnan(mod_subset)):
                contains_land[ii] = True
            # Check that neighbourhood contains a value
            if all(np.isnan(mod_subset)):
                pass
            else:
                # Create model and observation CDF objects
                mod_cdf = CDF(mod_subset, cdf_type = cdf_type)
                obs_cdf = CDF([obs_data[ii]], cdf_type = 'empirical')
            
                # Calculate CRPS and put into output array
                crps_list[ii] = mod_cdf.crps_fast(obs_data[ii])
                n_model_pts[ii] = int(mod_subset.shape[0])
                
    print("\r Complete.                             \n", end=" ", flush=True)

    return crps_list, n_model_pts, contains_land, mod_cdf, obs_cdf