'''
Python definitions used to help with plotting routines.

*Methods Overview*
    -> geo_scatter(): Geographical scatter plot.
'''

import matplotlib.pyplot as plt
from warnings import warn
from .logging_util import get_slug, debug, info, warn, error
import numpy as np

def ts_diagram(temperature, salinity, depth):
    
    fig = plt.figure(figsize = (10,7))
    ax = plt.scatter(salinity, temperature, c=depth)
    cbar = plt.colorbar()
    cbar.set_label('Depth (m)')
    plt.title('T-S Diagram')
    plt.xlabel('Salinity')
    plt.ylabel('Temperature')
    
    return fig, ax
            
def geo_scatter(longitude, latitude, c=None, s = None, 
                scatter_kwargs=None, coastline_kwargs=None,
                gridline_kwargs=None, figure_kwargs={},
                title="", figsize=None):
    '''
    Uses CartoPy to create a geographical scatter plot with land boundaries.
    
        Parameters
        ----------
        longitude : (array) Array of longitudes of marker locations
        latitude  : (array) Array of latitudes of marker locations
        colors    : (array) Array of values to use for colouring markers
        title     : (str) Plot title, to appear at top of figure
        xlim      : (tuple) Tuple of limits to apply to the x-axis (longitude axis)
        ylim      : (tuple) Limits to apply to the y-axis (latitude axis)
    
        Returns
        -------
        Figure and axis objects for further customisation
    
    '''
    try:
        import cartopy.crs as ccrs  # mapping plots
        from cartopy.feature import NaturalEarthFeature
    except ImportError:
        import sys
        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)
        
    if coastline_kwargs is None:
        coastline_kwargs = {'facecolor':[0.9,0.9,0.9], 'name':'coastline',
                            "scale":"50m"}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig = plt.figure(**figure_kwargs)

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    sca = ax.scatter(longitude, y=latitude, c=c, s=s, zorder = 100,
                     **scatter_kwargs)
    coast = NaturalEarthFeature(category='physical', **coastline_kwargs)
    ax.add_feature(coast, edgecolor='gray')
    #ax.coastlines(facecolor=[0.8,0.8,0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', linestyle='-')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.left_labels = True
    plt.title(title)
    
    if c is not None and 'vmax' in scatter_kwargs.keys() and 'vmin' in scatter_kwargs.keys():
        extend_max = np.nanmax(c) > scatter_kwargs['vmax']
        extend_min = np.nanmin(c) < scatter_kwargs['vmin']
        extend='neither'
        if extend_max and extend_min: extend="both"
        if extend_max and not extend_min: extend='max'
        if not extend_max and extend_min: extend='min'
    else:
        extend = 'neither'
    
    plt.colorbar(sca, extend=extend)
    ax.set_aspect('auto')

    plt.show()
    return fig, ax