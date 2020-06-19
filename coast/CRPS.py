import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension

class CRPS():
    '''
    Object for handling and storing necessary information, methods and outputs
    for calculation of Continuous Ranked Probability Score. The object is 
    initialized by passing it COAsT variables of model data, model domain and
    and observation object. CRPS can then be calculated using the 
    CRPS.calculate() function. This will return an array of CRPS values
    (if desired) or will store them inside the object. They can be accessed
    from the object by calling CRPS.crps or CRPS['crps']
    
    Example basic usage::
        
        $ crps_obj = coast.CRPS(nemo_data, nemo_domain, altimetry)
        $ crps_obj.calculate('sossheig', 'sla_filtered', nh_radius=111)
        $ crps_list = crps_obj.crps # Get crps values
        $ crps.map_plot() # Plots CRPS on map
    '''
    
    def __init__(self, mod_data, mod_dom, obs_object):
        """Initialisation of CRPS object.

        Args:
            mod_data (COAsT): COAsT model data object.
            mod_dom  (COAsT): COAsT model domain object.
            obs_object (OBSERVATION): COAsT OBSERVATION object.

        Returns:
            CRPS: Returns a new instance of a CRPS object.
        """
        
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
        self.n_model_pts = None
        self.contains_land = None
        self.longitude   = obs_object['longitude']
        self.latitude    = obs_object['latitude']
        return
        
    def __getitem__(self, varstr:str):
        """Gets variable from __dict__"""
        return self.__dict__[varstr]

    def calculate(self, mod_var: str, obs_var: str, nh_radius: float=111, 
                  nh_type: str="radius", cdf_type: str="empirical", 
                  time_interp:str="nearest"):
        """Calculate CRPS values for specified variables/methods/radii.

        Args:
            mod_var (str): Name of variable to use from model object.
            obs_var (str): Name of variable to use from observation object.
            nh_radius(float): Neighbourhood radius.
            nh_type(str): Neighbourhood type, either 'radius' or 'box'
            cdf_type(str): Type of CDF to use for model data.
                           Either 'empirical' or 'theoretical'
            time_interp(str): Type of time interpolation.
                              Either 'nearest' or 'linear'
        Returns:
            array: CRPS values.
        """
        self.mod_var = mod_var
        self.obs_var = obs_var
        self.nh_radius   = nh_radius
        self.nh_type     = nh_type
        self.cdf_type    = cdf_type
        self.time_interp = time_interp
        tmp = self.calculate_sonf()
        self.crps = tmp[0]
        self.n_model_pts = tmp[1]
        self.contains_land = tmp[2]
        return 
    
    def cdf_plot(self, index):
        index = [index]
        tmp = self.calculate_sonf(index)
        crps_tmp = tmp[0]
        n_mod_pts = tmp[1]
        contains_land = tmp[2]
        mod_cdf = tmp[3]
        obs_cdf = tmp[4]
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(mod_cdf.disc_x, mod_cdf.disc_y, c='k', 
                linestyle='--')
        ax.plot(obs_cdf.disc_x, obs_cdf.disc_y, linestyle='--')
        ax.fill_between(mod_cdf.disc_x, mod_cdf.disc_y, 
                        obs_cdf.disc_y, alpha=0.5)
        titlestr = 'CRPS = ' + str(round( crps_tmp[index[0]], 3)) + '\n'
        titlestr = titlestr + '# Model Points : ' + str(n_mod_pts[0]) + '  |  '
        titlestr = titlestr + 'Contains land : ' + str(bool(contains_land[0])) 
        plt.title(titlestr)
        plt.legend(['Model', 'Observations'])
        return fig, ax
    
    def map_plot(self, crps_var: str='crps'):
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
        plt.scatter(self.longitude, self.latitude, c=self[crps_var])
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
    
    def calculate_sonf(self, nh_indices=None):
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
        crps_list = np.zeros( n_nh )*np.nan
        n_model_pts = np.zeros( n_nh )*np.nan
        contains_land = np.zeros( n_nh , dtype=bool)
        
        # Time interpolation weights object
        weights = interpolate_along_dimension(mod_data[mod_var], 
                                              obs['time'], 'time_counter', 
                                              method = time_interp)
        
        if nh_indices is None:
            nh_indices = np.arange(0,n_nh)
        
        # Loop over neighbourhoods
        for ii in nh_indices:
        
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
            if any(np.isnan(mod_subset)):
                contains_land[ii] = True

            if mod_subset.shape[0] == 0:
                crps_list[ii] = np.nan
            else:
                # Create model and observation CDF objects
                mod_cdf = CDF(mod_subset, cdf_type=cdf_type)
                obs_cdf = CDF(obs[obs_var][ii], cdf_type='empirical')
                
                # Calculate CRPS and put into output array
                crps_list[ii] = mod_cdf.difference(obs_cdf)
                n_model_pts[ii] = mod_cdf.sample_size[0]

        return crps_list, n_model_pts, contains_land, mod_cdf, obs_cdf
    