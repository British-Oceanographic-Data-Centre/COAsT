'''
Python definitions used to help with plotting routines.

*Methods Overview*
    -> geo_scatter(): Geographical scatter plot.
'''

import matplotlib.pyplot as plt
from warnings import warn
from .logging_util import get_slug, debug, info, warn, error
import numpy as np

def create_geo_axes(lonbounds, latbounds):
    '''
    A routine for creating an axis for any geographical plot. Within the
    specified longitude and latitude bounds, a map will be drawn up using
    cartopy. Any type of matplotlib plot can then be added to this figure.
    For example:
        
    Example Useage
    #############
    
        f,a = create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.corr, 
                        vmin=.75, vmax=1,
                        edgecolors='k', linewidths=.5, zorder=100)
        f.colorbar(sca)
        a.set_title('SSH correlations \n Monthly PSMSL tide gauge vs CO9_AMM15p0', 
                    fontsize=9)
        
    * Note: For scatter plots, it is useful to set zorder = 100 (or similar
            positive number)
    '''

    import cartopy.crs as ccrs  # mapping plots
    from cartopy.feature import NaturalEarthFeature
        
    # If no figure or ax is provided, create a new one
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        
    coast = NaturalEarthFeature(category='physical', facecolor=[0.9,0.9,0.9], name='coastline',
                            scale='50m')
    ax.add_feature(coast, edgecolor='gray')
    #ax.coastlines(facecolor=[0.8,0.8,0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', linestyle='-')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.left_labels = True
    
    ax.set_xlim(lonbounds[0], lonbounds[1])
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.set_aspect('auto')

    plt.show()
    return fig, ax


def ts_diagram(temperature, salinity, depth, fig=None, ax=None):
    
    if fig is None:
        fig = plt.figure(figsize = (10,7))
    if ax is None:
        ax = plt.subplot(111)
    plt.scatter(salinity, temperature, c=depth)
    cbar = plt.colorbar()
    cbar.set_label('Depth (m)')
    plt.title('T-S Diagram')
    plt.xlabel('Salinity')
    plt.ylabel('Temperature')
    
    return fig, ax
            
def geo_scatter(longitude, latitude, c=None, s = None, fig = None, ax = None,
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
        
    # If no figure or ax is provided, create a new one
    if fig is None:
        fig = plt.figure(**figure_kwargs)
    if ax is None:
        ax = plt.subplot(111, projection=ccrs.PlateCarree())
        
    sca = ax.scatter(longitude, y=latitude, c=c, s=s,
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
    
    # Automatically determine if the colorbar needs to be extended.
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

def determine_colorbar_extension(color_data, vmin, vmax):
    ''' Can be used to automatically determine settings for colorbar 
    extension arrows. Color_data is the data used for the colormap, vmin
    and vmax are the colorbar limits. Will output a string: "both", "max",
    "min" or "neither", which can be inserted straight into a call to
    matplotlib.pyplot.colorbar().
    '''
    extend_max = np.nanmax(color_data) > vmax
    extend_min = np.nanmin(color_data) < vmin
    
    if extend_max and extend_min: return "both"
    elif extend_max and not extend_min: return 'max'
    elif not extend_max and extend_min: return 'min'
    else: return 'neither'

def determine_clim_by_standard_deviation(color_data, n_std_dev=2.5):
    ''' Automatically determine color limits based on number of standard
    deviations from the mean of the color data (color_data). Useful if there
    are outliers in the data causing difficulties in distinguishing most of 
    the data. Outputs vmin and vmax which can be passed to plotting routine
    or plt.clim().
    '''
    color_data_mean = np.nanmean(color_data)
    color_data_std = np.nanstd(color_data)
    vmin = color_data_mean - n_std_dev*color_data_std
    vmax = color_data_mean + n_std_dev*color_data_std
    return vmin, vmax