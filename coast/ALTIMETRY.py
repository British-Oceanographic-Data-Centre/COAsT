from .COAsT import COAsT
from .OBSERVATION import OBSERVATION
from warnings import warn
import numpy as np
import xarray as xa

class ALTIMETRY(OBSERVATION):

    def __init__(self):
        super()
        self.sla_filtered = None
        self.sla_unfiltered = None
        self.mdt = None
        self.ocean_tide = None
        self.latitude = None
        self.longitude = None
        self.time = None
        # List of variables that are actually in the object (successfully read)
        self.var_list = []
        # Mapping of quick access variables to dataset variables
        # {'referencing_var' : 'dataset_var'}.
        self.var_dict = {'sla_filtered'   : 'sla_filtered',
                         'sla_unfiltered' : 'sla_unfiltered',
                         'mdt'            : 'mdt',
                         'ocean_tide'     : 'ocean_tide',
                         'longitude'      : 'longitude', 
                         'latitude'       : 'latitude',
                         'time'           : 'time'}

    def set_command_variables(self):
        """
         A method to make accessing the following simpler
        """
        
        for key, value in self.var_dict.items():
            try:
                setattr( self, key, self.dataset[value] )
                self.var_list.append(key)
            except AttributeError as e:
                warn(str(e))
                
        self.adjust_longitudes()
        
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
        coast = NaturalEarthFeature(category='physical', scale='50m', facecolor='none', name='coastline')
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
