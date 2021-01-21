from .COAsT import COAsT  # ???
import numpy as np
import xarray as xr
from .logging_util import get_slug, debug, error, info
import sklearn.metrics as metrics
from . import general_utils, plot_util, crps_util

class ALTIMETRY(COAsT):
    '''
    An object for reading, storing and manipulating altimetry data.
    Currently the object contains functionality for reading altimetry netCDF 
    data from the CMEMS database. This is the default for initialisation.
    
    Data should be stored in an xarray.Dataset, in the form:
        
    * Date Format Overview *
        
        1. A single dimension (time).
        2. Three coordinates: time, latitude, longitude. All lie on the 't_dim'
           dimension.
        3. Observed variable DataArrays on the t_dim dimension.
        
    There are currently no naming conventions for the variables however
    examples from the CMEMS database include sla_filtered, sla_unfiltered and
    mdt (mean dynamic topography).
    
    * Methods Overview *
        
        *Initialisation and File Reading*
        -> __init__(): Initialises an ALTIMETRY object.
        -> read_cmems(): Reads data from a CMEMS netCDF file.
        
        *Plotting*
        -> quick_plot(): Makes a quick along-track plot of specified data.
        
        *Model Comparison*
        -> obs_operator(): For interpolating model data to this object.
        -> cprs(): Calculates the CRPS between a model and obs variable.
        -> difference(): Differences two specified variables
        -> absolute_error(): Absolute difference, two variables
        -> mean_absolute_error(): MAE between two variables
        -> root_mean_square_error(): RMSE between two variables
        -> time_mean(): Mean of a variable in time
        -> time_std(): St. Dev of a variable in time
        -> time_correlation(): Correlation between two variables
        -> time_covariance(): Covariance between two variables
        -> basic_stats(): Calculates multiple of the above metrics.

    '''
##############################################################################
###                ~ Initialisation and File Reading ~                     ###
##############################################################################

    def __init__(self, file=None, chunks: dict=None, multiple=False):
        debug(f"Creating a new {get_slug(self)}")
        if file is not None:
            self.read_cmems(file, chunks, multiple)
        else:
            self.dataset = None
        debug(f"{get_slug(self)} initialised")
        return
    
    def read_cmems(self, file, chunks, multiple):
        ''' Reads altimetry data from a CMEMS netcdf file. Calls COAsT.init()
        to make use of its load methods'''
        super().__init__(file, chunks, multiple)
        self.dataset = self.dataset.rename_dims(self.dim_mapping)
        #self.dataset.attrs = {}

    def set_dimension_mapping(self):
        self.dim_mapping = {'time': 't_dim'}
        debug(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")

    def set_variable_mapping(self):
        self.var_mapping = None
        debug(f"{get_slug(self)} var_mapping set to {self.var_mapping}")
        
    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        debug(f"Subsetting {get_slug(self)} indices in {lonbounds}, {latbounds}")
        lon = self.dataset.longitude.copy()
        lat = self.dataset.latitude
        lon[lon>180] = lon[lon>180] - 360
        lon[lon<-180] = lon[lon<-180] + 360
        ff1 = ( lon > lonbounds[0] ).astype(int)  # FIXME This should fail? We can just treat bools as ints here...
        ff2 = ( lon < lonbounds[1] ).astype(int)
        ff3 = ( lat > latbounds[0] ).astype(int)
        ff4 = ( lat < latbounds[1] ).astype(int)
        indices = np.where( ff1 * ff2 * ff3 * ff4 )
        return indices[0]

##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################

    def quick_plot(self, color_var_str: str=None):
        '''
        '''
        
        if color_var_str is not None:
            color_var = self.dataset[color_var_str]
            title = color_var_str
        else:
            color_var = None
            title = 'Altimetry observation locations'
        info("Drawing a quick plot...")
        fig, ax =  plot_util.geo_scatter(self.dataset.longitude, 
                                         self.dataset.latitude,
                                         c=color_var, title=title )
        info("Plot ready, displaying!")
        return fig, ax

##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################
    
    def obs_operator(self, model, mod_var_name:str, 
                                time_interp = 'nearest', model_mask=None):
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
        model_mask : Mask to apply to model data in geographical interpolation 
             of model. For example, use to ignore land points. 
             If None, no mask is applied. If 'bathy', model variable
             (bathymetry==0) is used. Custom 2D mask arrays can be
             supplied.
        Returns
        -------
        Adds a DataArray to self.dataset, containing interpolated values.
        '''

        debug(f"Interpolating {get_slug(model)} \"{mod_var_name}\" with time_interp \"{time_interp}\"")

        # Determine mask
        if model_mask=='bathy':
            model_mask = model.dataset.bathymetry.values==0
        
        # Get data arrays
        mod_var_array = model.dataset[mod_var_name]

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
    
    def crps(self, model_object, model_var_name, obs_var_name, 
             nh_radius: float = 20, cdf_type:str='empirical', 
             time_interp:str='linear', create_new_object = True):
        
        '''
        Comparison of observed variable to modelled using the Continuous
        Ranked Probability Score. This is done using this ALTIMETRY object.
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
        create_new_obj (bool): If True, save output to new ALTIMETRY obj.
                               Otherwise, save to this obj.
          
        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags
        
        Example Useage
        -------
        # Compare modelled 'sossheig' with 'sla_filtered' using CRPS
        crps = altimetry.crps(nemo, 'sossheig', 'sla_filtered')
        '''
        
        mod_var = model_object.dataset[model_var_name]
        obs_var = self.dataset[obs_var_name]
        
        crps_list, n_model_pts, contains_land = crps_util.crps_sonf_moving( 
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
    
    def difference(self, var_str0:str, var_str1:str, date0=None, date1=None):
        ''' Difference two variables defined by var_str0 and var_str1 between
        two dates date0 and date1. Returns xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        diff = var0 - var1
        return xr.DataArray(diff, dims='t_dim', name='error',
                            coords={'time':self.dataset.time})
    
    def absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Absolute difference two variables defined by var_str0 and var_str1 
        between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        adiff = np.abs(var0 - var1)
        return xr.DataArray(adiff, dims='t_dim', name='absolute_error', 
                            coords={'time':self.dataset.time})
    
    def mean_absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Mean absolute difference two variables defined by var_str0 and 
        var_str1 between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        mae = metrics.mean_absolute_error(var0, var1)
        return mae
    
    def root_mean_square_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Root mean square difference two variables defined by var_str0 and 
        var_str1 between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        rmse = metrics.mean_squared_error(var0, var1)
        return np.sqrt(rmse)
    
    def time_mean(self, var_str, date0=None, date1=None):
        ''' Time mean of variable var_str between dates date0, date1'''
        var = self.dataset[var_str]
        var = general_utils.dataarray_time_slice(var, date0, date1)
        return np.nanmean(var)
    
    def time_std(self, var_str, date0=None, date1=None):
        ''' Time st. dev of variable var_str between dates date0 and date1'''
        var = self.dataset[var_str]
        var = general_utils.dataarray_time_slice(var, date0, date1)
        return np.nanstd(var)
    
    def time_correlation(self, var_str0, var_str1, date0=None, date1=None, 
                         method='pearson'):
        ''' Time correlation between two variables defined by var_str0, 
        var_str1 between dates date0 and date1. Uses Pandas corr().'''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename('var1')
        var1 = var1.rename('var2')
        var0 = general_utils.dataarray_time_slice(var0, date0, date1)
        var1 = general_utils.dataarray_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        corr = pdvar.corr(method=method)
        return corr.iloc[0,1]
    
    def time_covariance(self, var_str0, var_str1, date0=None, date1=None):
        ''' Time covariance between two variables defined by var_str0, 
        var_str1 between dates date0 and date1. Uses Pandas corr().'''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename('var1')
        var1 = var1.rename('var2')
        var0 = general_utils.dataarray_time_slice(var0, date0, date1)
        var1 = general_utils.dataarray_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        cov = pdvar.cov()
        return cov.iloc[0,1]
    
    def basic_stats(self, var_str0, var_str1, date0 = None, date1 = None,
                    create_new_object = True):
        ''' Calculates a selection of statistics for two variables defined by
        var_str0 and var_str1, between dates date0 and date1. This will return
        their difference, absolute difference, mean absolute error, root mean 
        square error, correlation and covariance. If create_new_object is True
        then this method returns a new ALTIMETRY object containing statistics,
        otherwise variables are saved to the dateset inside this object. '''
        
        diff = self.difference(var_str0, var_str1, date0, date1)
        ae = self.absolute_error(var_str0, var_str1, date0, date1)
        mae = self.mean_absolute_error(var_str0, var_str1, date0, date1)
        rmse = self.root_mean_square_error(var_str0, var_str1, date0, date1)
        corr = self.time_correlation(var_str0, var_str1, date0, date1)
        cov = self.time_covariance(var_str0, var_str1, date0, date1)
        
        if create_new_object:
            new_object = ALTIMETRY()
            new_dataset = self.dataset[['longitude','latitude','time']]
            new_dataset['absolute_error'] = ae
            new_dataset['error'] = diff
            new_dataset['mae'] = mae
            new_dataset['rmse'] = rmse
            new_dataset['corr'] = corr
            new_dataset['cov'] = cov
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset['absolute_error'] = ae
            self.dataset['error'] = diff
            self.dataset['mae'] = mae
            self.dataset['rmse'] = rmse
            self.dataset['corr'] = corr
            self.dataset['cov'] = cov