"""
Python definitions used to help with plotting routines.

*Methods Overview*
    -> geo_scatter(): Geographical scatter plot.
"""

import matplotlib.pyplot as plt
from warnings import warn
from .logging_util import warn
import numpy as np


def r2_lin(x, y, fit):
    """For calculating r-squared of a linear fit. Fit should be a python polyfit object."""
    y_estimate = fit(x)
    difference = (y - y_estimate) ** 2
    y_mean = np.nanmean(y)
    mean_square_deviation = (y - y_mean) ** 2

    total_deviation = np.nansum(mean_square_deviation)
    residual = np.nansum(difference)

    correlation_coefficient = 1 - residual / total_deviation

    return correlation_coefficient


def scatter_with_fit(x, y, s=10, c="k", yex=True, dofit=True):
    """Does a scatter plot with a linear fit. Will also draw y=x for
    comparison.

    Parameters
    ----------
    x     : (array) Values for the x-axis
    y     : (array) Values for the y-axis
    s     : (float or array) Marker size(s)
    c     : (float or array) Marker colour(s)
    yex   : (bool) True to plot y=x
    dofit : (bool) True to calculate and plot linear fit

    Returns
    -------
    Figure and axis objects for further customisation

    Example Useage
    -------
    x = np.arange(0,50)
    y = np.arange(0,50)/1.5
    f,a = scatter_with_fit(x,y)
    a.set_title('Example scatter with fit')
    a.set_xlabel('Example x axis')
    a.set_ylabel('Example y axis')
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    combined_mask = np.ma.mask_or(x.mask, y.mask)
    x.mask = combined_mask
    y.mask = combined_mask

    xmax = np.ma.max(x)
    xmin = np.ma.min(x)
    ymax = np.ma.max(y)
    ymin = np.ma.min(y)
    axmax0 = np.max([xmax, ymax])
    axmin0 = np.min([xmin, ymin])
    axmin = axmin0 - 0.1 * np.abs(axmax0 - axmin0)
    axmax = axmax0 + 0.1 * np.abs(axmax0 - axmin0)

    if yex:
        line_x = [axmin, axmax]
        fit_yx = np.poly1d([1, 0])
        ax.plot(line_x, fit_yx(line_x), c=[0.5, 0.5, 0.5], linewidth=1)

    ax.scatter(x, y, c=c, s=s)

    if dofit:
        line_x = [axmin, axmax]
        # Calculate data fit and cast to poly1d object
        fit_tmp = np.ma.polyfit(x, y, 1)
        fit = np.poly1d(fit_tmp)
        ax.plot(line_x, fit(line_x), c=[1, 128 / 255, 0], linewidth=1.5)
        r2 = r2_lin(x, y, fit)

    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid()

    if dofit:
        ax.text(
            0.4, 0.125, "{} {:03.2f} {} {:03.2f}".format("y =", fit_tmp[0], "x +", fit_tmp[1]), transform=ax.transAxes
        )
        ax.text(0.4, 0.05, "{} {:03.2f} ".format("$R^2$ =", r2), transform=ax.transAxes)

    return fig, ax


def create_geo_subplots(lonbounds, latbounds, n_r=1, n_c=1, figsize=(7, 7)):
    """
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
    """

    import cartopy.crs as ccrs  # mapping plots
    from cartopy.feature import NaturalEarthFeature

    # If no figure or ax is provided, create a new one
    # fig = plt.figure()
    # fig.clf()
    fig, ax = plt.subplots(
        n_r, n_c, subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True, sharex=True, figsize=figsize
    )
    land_color = [0.9, 0.9, 0.9]
    coast_color = [0, 0, 0]
    coast_width = 0.25

    if n_r * n_c > 1:
        ax = ax.flatten()
        for rr in range(n_r * n_c):
            coast = NaturalEarthFeature(category="physical", facecolor=land_color, name="coastline", scale="50m")
            ax[rr].add_feature(coast, edgecolor=coast_color, linewidth=coast_width)
            # ax.coastlines(facecolor=[0.8,0.8,0.8])
            gl = ax[rr].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
            gl.top_labels = False
            gl.right_labels = False
            if rr % n_c == 0:
                gl.left_labels = True
            else:
                gl.left_labels = False

            if np.abs(n_r * n_c - rr) <= n_c:
                gl.bottom_labels = True
            else:
                gl.bottom_labels = False
            ax[rr].set_xlim(lonbounds[0], lonbounds[1])
            ax[rr].set_ylim(latbounds[0], latbounds[1])
            ax[rr].set_aspect("auto")

        ax = ax.reshape((n_r, n_c))
    else:
        coast = NaturalEarthFeature(category="physical", facecolor=land_color, name="coastline", scale="50m")
        ax.add_feature(coast, edgecolor=coast_color, linewidth=coast_width)
        # ax.coastlines(facecolor=[0.8,0.8,0.8])
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
        gl.top_labels = False
        gl.right_labels = False

        gl.left_labels = True
        gl.bottom_labels = True

        ax.set_xlim(lonbounds[0], lonbounds[1])
        ax.set_ylim(latbounds[0], latbounds[1])
        ax.set_aspect("auto")

    plt.show()
    return fig, ax


def create_geo_axes(lonbounds, latbounds):
    """
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
    """

    import cartopy.crs as ccrs  # mapping plots
    from cartopy.feature import NaturalEarthFeature

    # If no figure or ax is provided, create a new one
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    coast = NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="50m")
    ax.add_feature(coast, edgecolor="gray")
    # ax.coastlines(facecolor=[0.8,0.8,0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.left_labels = True

    ax.set_xlim(lonbounds[0], lonbounds[1])
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.set_aspect("auto")

    plt.show()
    return fig, ax


def ts_diagram(temperature, salinity, depth):

    fig = plt.figure(figsize=(10, 7))
    ax = plt.scatter(salinity, temperature, c=depth)
    cbar = plt.colorbar()
    cbar.set_label("Depth (m)")
    plt.title("T-S Diagram")
    plt.xlabel("Salinity")
    plt.ylabel("Temperature")

    return fig, ax


def geo_scatter(
    longitude,
    latitude,
    c=None,
    s=None,
    scatter_kwargs=None,
    coastline_kwargs=None,
    gridline_kwargs=None,
    figure_kwargs={},
    title="",
    figsize=None,
):  # TODO Some unused parameters here
    """
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

    """
    try:
        import cartopy.crs as ccrs  # mapping plots
        from cartopy.feature import NaturalEarthFeature
    except ImportError:
        import sys

        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)

    if coastline_kwargs is None:
        coastline_kwargs = {"facecolor": [0.9, 0.9, 0.9], "name": "coastline", "scale": "50m"}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig = plt.figure(**figure_kwargs)

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    sca = ax.scatter(longitude, y=latitude, c=c, s=s, zorder=100, **scatter_kwargs)
    coast = NaturalEarthFeature(category="physical", **coastline_kwargs)
    ax.add_feature(coast, edgecolor="gray")
    # ax.coastlines(facecolor=[0.8,0.8,0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.left_labels = True
    plt.title(title)

    if c is not None and "vmax" in scatter_kwargs.keys() and "vmin" in scatter_kwargs.keys():
        extend_max = np.nanmax(c) > scatter_kwargs["vmax"]
        extend_min = np.nanmin(c) < scatter_kwargs["vmin"]
        extend = "neither"
        if extend_max and extend_min:
            extend = "both"
        if extend_max and not extend_min:
            extend = "max"
        if not extend_max and extend_min:
            extend = "min"
    else:
        extend = "neither"

    plt.colorbar(sca, extend=extend)
    ax.set_aspect("auto")

    plt.show()
    return fig, ax


def determine_colorbar_extension(color_data, vmin, vmax):
    """Can be used to automatically determine settings for colorbar
    extension arrows. Color_data is the data used for the colormap, vmin
    and vmax are the colorbar limits. Will output a string: "both", "max",
    "min" or "neither", which can be inserted straight into a call to
    matplotlib.pyplot.colorbar().
    """
    extend_max = np.nanmax(color_data) > vmax
    extend_min = np.nanmin(color_data) < vmin

    if extend_max and extend_min:
        return "both"
    elif extend_max and not extend_min:
        return "max"
    elif not extend_max and extend_min:
        return "min"
    else:
        return "neither"


def determine_clim_by_standard_deviation(color_data, n_std_dev=2.5):
    """Automatically determine color limits based on number of standard
    deviations from the mean of the color data (color_data). Useful if there
    are outliers in the data causing difficulties in distinguishing most of
    the data. Outputs vmin and vmax which can be passed to plotting routine
    or plt.clim().
    """
    color_data_mean = np.nanmean(color_data)
    color_data_std = np.nanstd(color_data)
    vmin = color_data_mean - n_std_dev * color_data_std
    vmax = color_data_mean + n_std_dev * color_data_std
    return vmin, vmax
