import numpy as np
import xarray as xr
from warnings import warn
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
from .COAsT import COAsT

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
        $ crps_list = crps_obj.crps # Get crps values
        $ crps.map_plot() # Plots CRPS on map
    '''
    
###############################################################################
#######                       ~Initialisation~                          #######
###############################################################################
    
    def __init__(self, model: COAsT, observations: COAsT,
                 var_name_mod:str, var_name_obs:str, nh_radius: float=20, 
                 nh_type: str="radius", cdf_type: str="empirical", 
                 time_interp: str="nearest"):
        """Initialisation of CRPS object.

        Args:
            mod_data (COAsT): COAsT model data object.
            mod_dom  (COAsT): COAsT model domain object.
            obs_object (OBSERVATION): COAsT OBSERVATION object.
            mod_var (str): Name of variable to use from model object.
            obs_var (str): Name of variable to use from observation object.
            nh_radius(float): Neighbourhood radius.
            nh_type(str): Neighbourhood type, either 'radius' or 'box'
            cdf_type(str): Type of CDF to use for model data.
                           Either 'empirical' or 'theoretical'
            time_interp(str): Type of time interpolation.
                              Either 'nearest' or 'linear'

        Returns:
            CRPS: Returns a new instance of a CRPS object.
        """
        #Input variables
        self.dataset      = observations[['longitude','latitude','time']]
        self.model        = model.dataset
        self.observations = observations.dataset
        self.nh_radius = nh_radius
        self.nh_type = nh_type
        self.cdf_type = cdf_type
        self.time_interp = time_interp
        self.var_name_mod = var_name_mod
        self.var_name_obs = var_name_obs
        self.dataset.attrs = {'title':'Continous Ranked Probability Score ' +
                              'for model('+ var_name_mod + 
                              ') vs observations(' + var_name_obs + '): '}
        # Output variables
        self.calculate()
        return
    
###############################################################################
#######                       ~General Routines~                        #######
###############################################################################
    
    def __getitem__(self, varstr:str):
        """Gets variable from __dict__"""
        return self.__dict__[varstr]

    def calculate(self):
        """Calculate CRPS values for specified variables/methods/radii."""
        tmp = self.calculate_sonf(self.model[self.var_name_mod], 
                                  self.observations[self.var_name_obs], 
                                  self.nh_radius, self.nh_type,
                                  self.cdf_type, self.time_interp)
        self.dataset['crps'] = tmp[0]
        self.dataset['n_model_pts']= tmp[1]
        self.dataset['contains_land'] = tmp[2]
        self.dataset['mean'] = np.nanmean(tmp[0])
        self.dataset['mean_noland'] = np.nanmean(tmp[0][tmp[2]==0])
        return 
    
    def calculate_sonf(self, model_data, obs_data, nh_radius: float, 
                       nh_type: str, cdf_type:str, time_interp:str):
        """Calculatues the Continuous Ranked Probability Score (CRPS)

        Calculatues the Continuous Ranked Probability Score (CRPS) using
        a single-observation and neighbourhood forecast (SONF). The statistic
        uses a comparison between the probability distributions of a model 
        neighbourhood subset and a single observation. The CRPS is calculated 
        independently for each observation. 
        """
        # Get relevant data and rename time dimension for interpolation
        model_data = model_data.rename({'t_dim':'time'})
        
        # Extract only x_dim, y_dim and t_dim dimensions. 
        # In case of other dimensions (e.g. depth), take first index
        for dim in model_data.dims:
            if dim not in ['x_dim', 'y_dim', 'time']:
                model_data = model_data.isel(dim=0)
    
        # Define output arrays
        n_neighbourhoods = obs_data.shape[0] 
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
    
    def subset_indices_by_distance(self, longitude, latitude,
                                   centre_lon: float, centre_lat: float, 
                                   radius: float):
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

        dist = self.calculate_haversine_distance(centre_lon, centre_lat, 
                                                 longitude, latitude)
        indices_bool = dist < radius
        indices = np.where(indices_bool.compute())

        return xr.DataArray(indices[0]), xr.DataArray(indices[1])
    
    def calculate_haversine_distance(self, lon1, lat1, lon2, lat2):
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
        distance = xr.ufuncs.sin(dlat) ** 2 + xr.ufuncs.cos(lat1) * xr.ufuncs.cos(lat2) * xr.ufuncs.sin(dlon) ** 2
        distance = 2 * 6371.007176 * xr.ufuncs.arcsin(xr.ufuncs.sqrt(distance))

        return distance
    
    
###############################################################################
#######                       ~PLOTTING ROUTINES~                       #######
###############################################################################
    
    def cdf_plot(self, index):
        """A comparison plot of the model and observation CDFs.

        Args:
            index (int): Observation index to plot CDFs for (single index).

        Returns:
            Figure and axes objects for the resulting image.
        """
        index=[index]
        tmp = self.calculate_sonf(self.model[self.var_name_mod], 
                                  self.observations[self.var_name_obs][index],
                                  self.nh_radius, self.nh_type, self.cdf_type,
                                  self.time_interp)
        crps_tmp = tmp[0]
        n_mod_pts = tmp[1]
        contains_land = tmp[2]
        mod_cdf = tmp[3]
        obs_cdf = tmp[4]
        fig, ax = mod_cdf.diff_plot(obs_cdf)
        titlestr = 'CRPS = ' + str(round( crps_tmp[0], 3)) + '\n'
        titlestr = titlestr + '# Model Points : ' + str(n_mod_pts[0]) + '  |  '
        titlestr = titlestr + 'Contains land : ' + str(bool(contains_land[0]))
        ax.set_title(titlestr)
        ax.grid()
        ax.legend(['Model', 'Observations'])
        return fig, ax

    def map_plot(self, stats_var: str='crps'):
        """Plots CRPS (or other variables) at observation locations on a map

        Args:
            stats_var (str): Either 'crps', 'n_model_pts' or 'contains_land'.

        Returns:
            Figure and axes objects for the resulting image.
        """
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
        plt.scatter(self.dataset.longitude, self.dataset.latitude, 
                    c=self.dataset[stats_var])
        plt.colorbar()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        coast = NaturalEarthFeature(category='physical', scale='50m',
                                    facecolor='none', name='coastline')
        ax.add_feature(coast, edgecolor='gray')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.bottom_labels = True
        gl.right_labels = False
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        title_dict = {'crps': 'Continuous Rank Probability Score',
                      'n_model_pts' : 'Number of Model Points Used',
                      'contains_land': 'CRPS Values that contain land'}
        try:
            plt.title(title_dict[stats_var])
        except:
            pass

        plt.show()
        return fig, ax
