import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
import scipy.interpolate as interpolate

class ERRORS():
    
    def __init__(self, mod_data, mod_dom, obs_object, 
                 mod_var:str, obs_var:str, space_interp: str='nearest',
                 time_interp: str='nearest'):
        self.mod_data = mod_data
        self.mod_dom = mod_dom
        self.obs_object = obs_object
        self.mod_var = mod_var
        self.obs_var = obs_var
        self.space_interp = space_interp
        self.time_interp = time_interp
        self.longitude   = obs_object['longitude']
        self.latitude    = obs_object['latitude']
        self.err = None
        self.abs_err = None
        self.corr = None
        self.mean_err = None
        self.mae = None
        self.rmse = None
        self.calculate(mod_data[mod_var], mod_dom, obs_object[obs_var], 
                       space_interp, time_interp)
        return
        
    def calculate(self,mod_var, mod_dom, obs_var, space_interp, time_interp):
        
        interpolated = np.zeros(len(obs_var))*np.nan
        # Get time interpolation weights
        time_weights = interpolate_along_dimension(mod_var, obs_var, 
                                                   'time_counter', 
                                              method = time_interp)
        for ii in range(0,len(interpolated)):
            x = np.array(mod_dom['nav_lon']).flatten()
            y = np.array(mod_dom['nav_lat']).flatten()
            d = np.array(time_weights[ii]).flatten()
            tmp = interpolate.griddata((x,y),d,(obs_var['longitude'][ii], 
                                                obs_var['latitude'][ii]))
            interpolated[ii] = tmp
        
        err = interpolated - obs_var
        abs_err = np.abs(err)
        corr = None
        mean_err = np.nanmean(err)
        mae = np.nanmean(abs_err)
        rmse = None
        
        return err, abs_err, corr, mean_err, mae, rmse
    
    def map_plot(self, stats_var: str='mae'):
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
        plt.scatter(self.longitude, self.latitude, c=self[stats_var])
        plt.colorbar()
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
        title_dict = {'err': 'Errors',
                      'abs_err' : 'Absolute Errors'}
        try:
            plt.title(title_dict[stats_var])
        except:
            pass
        
        plt.show()
        return fig, ax