from .COAsT import COAsT
from .OBSERVATION import OBSERVATION
from warnings import warn
import numpy as np
import xarray as xr

class ALTIMETRY(OBSERVATION):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_type = 'moving'
        return
    
    def set_dimension_mapping(self):
        self.dim_mapping = None
        

    def set_variable_mapping(self):
        self.var_mapping = None
        
    def quick_plot(self, var: str=None):
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        except ImportError:
            import sys
            warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
            sys.exit(-1)
            
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        if var is None:
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

        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_right = False
        gl.ylabels_left = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        plt.show()
        return fig, ax