import numpy as np
import xarray as xa
from warnings import warn
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension

class CRPS():
    
    def __init__(self, mod_data, mod_dom, obs_object):
        '''
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
        '''
        self.mod_data    = mod_data
        self.mod_dom     = mod_dom
        self.obs_object  = obs_object
        self.nh_radius   = None
        self.nh_type     = None
        self.cdf_type    = None
        self.time_interp = None
        self.crps        = None
        self.mod_var     = None
        self.obs_var     = None
        self.longitude   = obs_object['longitude']
        self.latitude    = obs_object['latitude']
        return
        
    def __getitem__(self, indices):
        return self.crps[indices]

    
    def calculate(self, mod_var: str, obs_var: str, nh_radius: float=111, 
                  nh_type: str="radius", cdf_type: str="empirical", 
                  time_interp:str="nearest"):
        self.mod_var = mod_var
        self.obs_var = obs_var
        self.nh_radius   = nh_radius
        self.nh_type     = nh_type
        self.cdf_type    = cdf_type
        self.time_interp = time_interp
        crps_local = self.calculate_sonf()
        self.crps = crps_local
        return crps_local
    
    def cdf_plot(self, index):
        
        #plt.figure()
        #ax = plt.subplot(111)
        #ax.plot(mod_cdf.disc_x, mod_cdf.disc_y, c='k', 
        #        linestyle='--')
        #ax.plot(obs_cdf.disc_x, obs_cdf.disc_y, linestyle='--')
        #ax.fill_between(mod_cdf.disc_x, mod_cdf.disc_y, 
        #                obs_cdf.disc_y, alpha=0.5)
        #plt.title(round( crps_list[ii], 3))
        return
    
    def map_plot(self):
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
            import matplotlib.pyplot as plt
        except ImportError:
            import sys
            warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
            sys.exit(-1)
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.scatter(self.longitude, self.latitude, c=self.crps)
        plt.colorbar()
        plt.title('Continuous Rank Probability Score')
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        coast = NaturalEarthFeature(category='physical', scale='50m', 
                                    facecolor='none', name='coastline')
        ax.add_feature(coast, edgecolor='gray')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_right = False
        gl.ylabels_left = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.show()
        return fig, ax
    
    def calculate_sonf(self):
        """Calculatues the Continuous Ranked Probability Score (CRPS)
    
        Calculatues the Continuous Ranked Probability Score (CRPS) using
        a single-observation and neighbourhood forecast (SONF). The statistic
        uses a comparison between the probability distributions of a model 
        neighbourhood subset and a single observation. The CRPS is calculated 
        independently for each observation. 

        Keyword arguments:
        nemo_var_name -- COAsT variable string.
        
        
        return: Array of CRPS scores for each observation supplied.
        """
        
        # Define var_dict to determine which variable to use and define some
        # function variables
        mod_data     = self.mod_data
        obs          = self.obs_object
        mod_dom      = self.mod_dom
        time_interp  = self.time_interp
        nh_type      = self.nh_type
        nh_radius    = self.nh_radius
        cdf_type     = self.cdf_type
        mod_var      = self.mod_var
        obs_var      = self.obs_var
        if len(mod_data[mod_var].dims) > 3:
            raise Exception('COAsT: CRPS Input data must only have dims ' + 
                            '(time, lon, lat)')
    
        # Define output array and check for scalars being used as observations.
        # If so, put obs into lists/arrays.
        n_nh = obs[obs_var].shape[0] # Number of neighbourhoods (n_nh)  
        crps_list = np.zeros( n_nh )
        
        # Time interpolation weights object
        weights = interpolate_along_dimension(mod_data[mod_var], 
                                              obs['time'], 'time_counter', 
                                              method = time_interp)
        # Loop over neighbourhoods
        for ii in range(0, n_nh):
        
            # Neighbourhood centre
            cntr_lon = self.longitude[ii]
            cntr_lat = self.latitude[ii]
        
            # Get model neighbourhood subset using specified method
            if nh_type == "radius":
                subset_indices = mod_dom.subset_indices_by_distance(cntr_lon, 
                                 cntr_lat, nh_radius)
            elif nh_type == "box":
                pass
            
            # Subset model data in time and space: What is the model doing at
            # observation times?
            mod_subset = weights[ii][xa.DataArray(subset_indices[0]), 
                                     xa.DataArray(subset_indices[1])]   

            if mod_subset.shape[0] == 0:
                crps_list[ii] = np.nan
                #raise Exception('COAsT: CRPS model neighbourhood contains no' +
                #                ' points. Try increasing neighbourhood size.')
            else:
                # Create model and observation CDF objects
                mod_cdf = CDF(mod_subset, cdf_type=cdf_type)
                obs_cdf = CDF(obs[obs_var], cdf_type=cdf_type)
                
                # Calculate CRPS and put into output array
                crps_list[ii] = mod_cdf.difference(obs_cdf)
    
        return crps_list
    