from .COAsT import COAsT  # ???
from .OBSERVATION import OBSERVATION
import numpy as np
import xarray as xr
from .logging_util import get_slug, debug, error, info


class ALTIMETRY(OBSERVATION):
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

    def __init__(self, *args, **kwargs):
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(*args, **kwargs)
        self.observation_type = 'moving'
        self.dataset = self.dataset.rename_dims(self.dim_mapping)
        debug(f"{get_slug(self)} initialised")

    def set_dimension_mapping(self):
        self.dim_mapping = {'time': 't_dim'}
        debug(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")

    def set_variable_mapping(self):
        self.var_mapping = None
        debug(f"{get_slug(self)} var_mapping set to {self.var_mapping}")

    def quick_plot(self, var: str = None):
        '''
        Quick geographical plot of altimetry data for a specified variable
    
        Example usage:
        --------------
        # Have a quick look at sla_filtered
        altimetry.quick_plot('sla_filtered')

        Parameters
        ----------
        var (str) : Variable to plot. Default is None, in which case only
            locations are plotted.
        Returns
        -------
        Figure and axes objects
        '''
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        except ImportError:
            error("CartoPy is not installed. Raising exception...")
            raise ImportError(
                "CartoPy not found - please run:\n"
                "conda install -c conda-forge cartopy\n"
                "OR\n"
                "pip install Cartopy"
            ) from None
        import matplotlib.pyplot as plt

        info("Drawing a quick plot...")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

        if var is None:  # TODO Variable cset appears to be unused
            cset = self.dataset.plot.scatter(x='longitude',y='latitude')
        else:
            cset = self.dataset.plot.scatter(x='longitude',y='latitude',hue=var)

        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        coast = NaturalEarthFeature(category='physical', scale='50m',
                                    facecolor=[0.8,0.8,0.8], name='coastline',
                                    alpha=0.5)
        ax.add_feature(coast, edgecolor='gray')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='-')

        gl.top_labels = False
        gl.bottom_labels = True
        gl.right_labels = False
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        info("Plot ready, displaying!")
        plt.show()
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

        debug(f"Interpolating {get_slug(model)} \"{mod_var_name}\" with time_interp \"{time_interp}\"")

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
        return  # TODO Should this return something? If not, the statement is not needed
