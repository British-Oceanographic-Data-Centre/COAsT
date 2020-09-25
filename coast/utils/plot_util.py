'''
Python definitions used to help with plotting routines.

*Methods Overview*
    -> geo_scatter(): Geographical scatter plot.
'''

import matplotlib.pyplot as plt
from warnings import warn
from .logging_util import get_slug, debug, info, warn, error
            
def geo_scatter(longitude, latitude, colors=None, 
                title='', xlim=None, ylim=None):
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
        import cartopy.feature  # add rivers, regional boundaries etc
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
        from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
    except ImportError:
        import sys
        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if colors is None:
        plt.scatter(longitude, latitude)
    else:
        plt.scatter(longitude, y=latitude, c = colors)

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
    plt.title(title)
    plt.colorbar()
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.show()
    return fig, ax