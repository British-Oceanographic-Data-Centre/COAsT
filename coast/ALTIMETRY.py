import numpy as np
import xarray as xr
from .COAsT import COAsT

class ALTIMETRY(COAsT):
    '''
    An object for reading, storing and manipulating altimetry data.
    Currently the objecgt is set up for reading altimetry netCDF data from
    the CMEMS database.
    
    Data should be stored in an xarray.Dataset, in the form:
        
    * Date Format Overview *
        
        1. A single dimension (time).
        2. Three coordinates: time, latitude, longitude. All lie on the time
           dimension.
        3. Observed variable DataArrays on the time dimension.
        
    There are currently no naming conventions for the variables however
    examples from the CMEMS database include sla_filtered, sla_unfiltered and
    mdt (mean dynamic topography).
    
    * Methods Overview *
    
    1. __init__(): Initialises an ALTIMETRY object.
    2. quick_plot(): Makes a quick plot of the data inside the object. 
    3. obs_operator(): For interpolating model data to this object.
    '''

    def __init__(self, file=None, chunks: dict=None, multiple=False):
        if file is not None:
            self.read_cmems(file, chunks, multiple)
        else:
            self.dataset = None
        return
    
    def read_cmems(self, file, chunks, multiple):
        super().__init__(file, chunks, multiple)
        self.dataset = self.dataset.rename_dims(self.dim_mapping)
        self.dataset.attrs = {}
        return

    def set_dimension_mapping(self):
        self.dim_mapping = {'time':'t_dim'}

    def set_variable_mapping(self):
        self.var_mapping = None

    def quick_plot(self, color_var_str: str=None):
        '''
        '''
        from .utils import plot_util
        
        if color_var_str is not None:
            color_var = self.dataset[color_var_str]
            title = color_var_str
        else:
            color_var = None
            title = 'Altimetry observation locations'
        
        fig, ax =  plot_util.geo_scatter(self.dataset.longitude, 
                                         self.dataset.latitude,
                                         color_var, title=title )
        return fig, ax
    
    def obs_operator(self, model, mod_var_name:str, 
                                time_interp = 'nearest'):
        '''
        For interpolating a model dataarray onto altimetry locations and times.
        
        For ALTIMETRY, the interpolation is done independently in two steps:
            1. Horizontal space
            2. Time
        Model data is taken at the surface if necessary (0 index). 
    
        Example usage:
        --------------
        altimetry.obs_operator(nemo_obj, 'sossheig')

        Parameters
        ----------
        model : model object (e.g. NEMO)
        mod_var: variable name string to use from model object
        time_interp: time interpolation method (optional, default: 'nearest')
            This can take any string scipy.interpolate would take. e.g.
            'nearest', 'linear' or 'cubic'
        Returns
        -------
        Adds a DataArray to self.dataset, containing interpolated values.
        '''
        
        # Get data arrays
        mod_var = model.dataset[mod_var_name]
        
        # Depth interpolation -> for now just take 0 index
        if 'z_dim' in mod_var.dims:
            mod_var = mod_var.isel(z_dim=0).squeeze()
        
        # Cast lat/lon to numpy arrays
        obs_lon = np.array(self.dataset.longitude).flatten()
        obs_lat = np.array(self.dataset.latitude).flatten()
        
        interpolated = model.interpolate_in_space(mod_var, obs_lon, 
                                                        obs_lat)
        
        # Interpolate in time if t_dim exists in model array
        if 't_dim' in mod_var.dims:
            interpolated = model.interpolate_in_time(interpolated, 
                                                     self.dataset.time,
                                                     interp_method=time_interp)
            # Take diagonal from interpolated array (which contains too many points)
            diag_len = interpolated.shape[0]
            diag_ind = xr.DataArray(np.arange(0, diag_len))
            interpolated = interpolated.isel(interp_dim=diag_ind, t_dim=diag_ind)
            interpolated = interpolated.swap_dims({'dim_0':'t_dim'})

        # Store interpolated array in dataset
        new_var_name = 'interp_' + mod_var_name
        self.dataset[new_var_name] = interpolated
        return
    
    def crps(self, model_object, model_var_name, obs_var_name, 
             nh_radius: float = 20, cdf_type:str='empirical', 
             time_interp:str='linear', create_new_object = True):
        
        '''
        Comparison of observed variable to modelled using the Continuous
        Ranked Probability Score. This is done using this TIDEGAUGE object.
        This method specifically performs a single-observation neighbourhood-
        forecast method.
        
        Parameters
        ----------
        model_object (model) : Model object (NEMO) containing model data
        model_var_name (str) : Name of model variable to compare.
        obs_var_name (str)   : Name of observed variable to compare.
        nh_radius (float)    : Neighbourhood rad
        cdf_type (str)       : Type of cumulative distribution to use for the
                               model data ('empirical' or 'theoretical').
                               Observations always use empirical.
        time_interp (str)    : Type of time interpolation to use (s)
        create_new_obj (bool):
          
        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags
        
        Example Useage
        -------
        # Compare modelled 'sossheig' with 'sla_filtered' using CRPS
        crps = altimetry.crps(nemo, 'sossheig', 'sla_filtered')
        '''
        
        from .utils import CRPS as crps
        
        mod_var = model_object.dataset[model_var_name]
        obs_var = self.dataset[obs_var_name]
        
        crps_list, n_model_pts, contains_land = crps.crps_sonf_moving( 
                               mod_var, 
                               obs_var.longitude.values, 
                               obs_var.latitude.values, 
                               obs_var.values, 
                               obs_var.time.values, 
                               nh_radius, cdf_type, time_interp )
        if create_new_object:
            new_object = ALTIMETRY()
            new_dataset = self.dataset[['longitude','latitude','time']]
            new_dataset['crps'] =  (('t_dim'),crps_list)
            new_dataset['crps_n_model_pts'] = (('t_dim'), n_model_pts)
            new_dataset['crps_contains_land'] = (('t_dim'), contains_land)
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset['crps'] =  (('t_dim'),crps_list)
            self.dataset['crps_n_model_pts'] = (('t_dim'), n_model_pts)
            self.dataset['crps_contains_land'] = (('t_dim'), contains_land)
    
    # def cdf_plot(self, index):
    #     """A comparison plot of the model and observation CDFs.
    
    #     Args:
    #         index (int): Observation index to plot CDFs for (single index).
    
    #     Returns:
    #         Figure and axes objects for the resulting image.
    #     """
    #     index=[index]
    #     tmp = self.calculate_sonf(self.model[self.var_name_mod], 
    #                               self.observations[self.var_name_obs][index],
    #                               self.nh_radius, self.nh_type, self.cdf_type,
    #                               self.time_interp)
    #     crps_tmp = tmp[0]
    #     n_mod_pts = tmp[1]
    #     contains_land = tmp[2]
    #     mod_cdf = tmp[3]
    #     obs_cdf = tmp[4]
    #     fig, ax = mod_cdf.diff_plot(obs_cdf)
    #     titlestr = 'CRPS = ' + str(round( crps_tmp[0], 3)) + '\n'
    #     titlestr = titlestr + '# Model Points : ' + str(n_mod_pts[0]) + '  |  '
    #     titlestr = titlestr + 'Contains land : ' + str(bool(contains_land[0]))
    #     ax.set_title(titlestr)
    #     ax.grid()
    #     ax.legend(['Model', 'Observations'])
    # return fig, ax
