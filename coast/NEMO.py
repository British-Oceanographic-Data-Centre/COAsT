from .COAsT import COAsT
import xarray as xa
import numpy as np
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def set_dimension_mapping(self):
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim', 
                            'y':'y_dim', 'x':'x_dim'}

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller

    def crps_sonf(self, mod_var_name, mod_dom, obs_object, obs_var_name,
                  nh_radius=111, nh_type = "radius", cdf_type = "empirical",
                  time_interp = "nearest", plot=False):
        """Calculatues the Continuous Ranked Probability Score (CRPS)
    
        Calculatues the Continuous Ranked Probability Score (CRPS) using
        a single-observation and neighbourhood forecast (SONF). The statistic
        uses a comparison between the probability distributions of a model 
        neighbourhood subset and a single observation. The CRPS is calculated 
        independently for each observation. 

        Keyword arguments:
        nemo_var_name -- COAsT variable string.
        nemo_dom -- COAsT DOMAIN object
        obs_lon -- Array of observation longitudes
        obs_lat -- Array of observation latitudes
        obs_var -- Array of observation variables
        nh_radius -- Neighbourhood radius in km (if radius method) or degrees 
                     (if box method).
        nh_type -- Neighbourhood determination method: 'radius' or 'box'.
        cdf_type -- Method for model CDF determination: 'empirical' or 
                    'theoretical'. Observation CDFs are always determined 
                    empirically.
        time_interp -- Method for interpolating in time, currently only
                       "none" and "nearest". For none, only a single model
                       time slice should be supplied and observations are
                       assumed to correspond with the slice correctly.
        plot -- True or False. Will plot up to five CDF comparisons and CRPS.
        
        return: Array of CRPS scores for each observation supplied.
        """
        # Define var_dict to determine which variable to use and define some
        # function variables
        mod_var = self.dataset[mod_var_name]
        if len(mod_var.dims) > 3:
            raise Exception('COAsT: CRPS Input data must only have dims ' + 
                            '(time, lon, lat)')
        obs_var = obs_object.dataset[obs_var_name]
        obs_lon = obs_object.dataset.longitude
        obs_lat = obs_object.dataset.latitude
        obs_time = obs_object.dataset.time
    
        # Define output array and check for scalars being used as observations.
        # If so, put obs into lists/arrays.
        n_nh = obs_lon.shape # Number of neighbourhoods (n_nh)  
        if len(n_nh) == 0: #Scalar case
            n_nh = 1
            obs_lon  =  [obs_lon]
            obs_lat  =  [obs_lat]
            obs_var  =  [obs_var]
            obs_time =  [obs_time]
        else:
            n_nh = n_nh[0]
        crps_list = np.zeros( n_nh )
        
        # Time interpolation weights object
        weights = interpolate_along_dimension(mod_var, obs_time, 
                                         'time_counter', method = time_interp)
        
        # Loop over neighbourhoods
        for ii in range(0, n_nh):
        
            # Neighbourhood centre
            cntr_lon = obs_lon[ii]
            cntr_lat = obs_lat[ii]
        
            # Get model neighbourhood subset using specified method
            if nh_type == "radius":
                subset_indices = mod_dom.subset_indices_by_distance(cntr_lon, 
                                 cntr_lat, nh_radius)
                
            elif nh_type == "box":
                lonbounds = [ cntr_lon - nh_radius, cntr_lon + nh_radius ]
                latbounds = [ cntr_lat - nh_radius, cntr_lat + nh_radius ]
                subset_indices = mod_dom.subset_indices_lonlat_box(lonbounds, 
                                                                    latbounds )   
            # Subset model data in time and space: What is the model doing at
            # observation times?
            mod_var_subset = weights[ii][xa.DataArray(subset_indices[0]), 
                                     xa.DataArray(subset_indices[1])]   

            if mod_var_subset.shape[0] == 0:
                raise Exception('COAsT: CRPS model neighbourhood contains no' +
                                ' points. Try increasing neighbourhood size.')
                
            # Create model and observation CDF objects
            mod_cdf = CDF(mod_var_subset, cdf_type=cdf_type)
            obs_cdf = CDF(obs_var[ii], cdf_type=cdf_type)
            
            # Calculate CRPS and put into output array
            crps_list[ii] = mod_cdf.difference(obs_cdf)
            
            if plot and n_nh<5:
                plt.figure()
                ax = plt.subplot(111)
                ax.plot(mod_cdf.disc_x, mod_cdf.disc_y, c='k', 
                        linestyle='--')
                ax.plot(obs_cdf.disc_x, obs_cdf.disc_y, linestyle='--')
                ax.fill_between(mod_cdf.disc_x, mod_cdf.disc_y, 
                                obs_cdf.disc_y, alpha=0.5)
                plt.title(round( crps_list[ii], 3))
    
        return crps_list
    
    