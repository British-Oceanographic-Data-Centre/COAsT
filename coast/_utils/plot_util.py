# pylint: disable=E0633
# Pylint add in order to remove a false positive with pyproj.Transform
"""
Python definitions used to help with plotting routines.

*Methods Overview*
    -> geo_scatter(): Geographical scatter plot.
"""

import sys
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy.interpolate as si
from tqdm import tqdm

from .logging_util import warn


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
        ax.text(0.4, 0.125, f"y = {fit_tmp[0]:03.2f} x + {fit_tmp[1]:03.2f}", transform=ax.transAxes)
        ax.text(0.4, 0.05, f"$R^2$ = {r2:03.2f}", transform=ax.transAxes)

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

    try:
        import cartopy.crs as ccrs  # mapping plots
        from cartopy.feature import NaturalEarthFeature
    except ImportError:
        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)
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
            coast = NaturalEarthFeature(category="physical", facecolor=land_color, name="coastline", scale="110m")
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
        coast = NaturalEarthFeature(category="physical", facecolor=land_color, name="coastline", scale="110m")
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

    try:
        import cartopy.crs as ccrs  # mapping plots
        from cartopy.feature import NaturalEarthFeature
    except ImportError:
        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)

    # If no figure or ax is provided, create a new one
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    coast = NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="110m")
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
    """
    A routine for plotting a T-S diagram.

    Args:
        temperature (array): temperature data
        salinity (array): salinity data
        depth (array): depth data

    Returns:
        fig, ax: fig and ax matplotlib elements
    """
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
    figure_kwargs={},
    title="",
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


def velocity_polar_bug_fix(u_velocity, v_velocity, latitude):
    """Adjust u and v to work-around a bug in cartopy=0.21.1 for quiver plotting
    specifically when using a stereographic projection. The bug means that
    the u component (x direction) of quivers will not be correctly
    proportioned relative to the v component (y direction). This function
    scales the u and v components correctly for plotting. BUT IT TO BE USED ONLY FOR PLOTTING.
    NOTE: only use this for cartopy maps with NorthPolarStereo or SouthPolarStereo
    projection.
    NOTE: this was developed with a bug that existed in cartopy=0.21.1

    Args:
        u_velocity (array): eastward velocity vectors
        v_velocity (array): northward velocity vectors
        latitude (array): latitude of the points in the same format as the u
        and v velocities.

    Returns:
        array, array: u_velocity and v_velocity that have been "corrected" to enable
        plotting in cartopy.
    """
    u_src_crs = u_velocity / np.cos(latitude / 180 * np.pi)
    v_src_crs = v_velocity * 1
    magnitude = (u_velocity**2 + v_velocity**2) ** 0.5
    magn_src_crs = (u_src_crs**2 + v_src_crs**2) ** 0.5
    u_new = u_src_crs * (magnitude / magn_src_crs)
    v_new = v_src_crs * (magnitude / magn_src_crs)
    return u_new, v_new


def make_projection(x_origin, y_origin):
    """Define projection centred on a given longitude, latitude point (WGS84) in meters.

    Args:
        x_origin (float): longitude of the centre point of the projection
        y_origin (float): latitude of the centre point of the projection

    Returns:
        CRS Object: A CRS for the bespoke projection defined here.
    """
    aeqd = pyproj.CRS(proj="aeqd", lon_0=x_origin, lat_0=y_origin, ellps="WGS84")
    # eqd = ccrs.CRS("+proj=aeqd +lon_0={:}".format(x_origin) +
    #                " +lat_0={:}".format(y_origin) + " +ellps=WGS84")
    return aeqd


def velocity_rotate(u_velocity, v_velocity, angle, to_north=True):
    """A function to change the direction of velocity components by a
    given angle

    Args:
        u_velocity (array): x-direction velocities along grid lines
        v_velocity (array): y-direction velocities along grid lines
        angle (array): angle of the rotation in degrees
        to_north (bool, optional): If True rotate with angle clockwise
        from 12 o'clock. If False rotate with angle anticlockwise from
        3 o'clock. Defaults to True.

    Returns:
        array, array: u and v velocities that have been rotated by
        the given angle
    """
    # use compass directions
    speed = (u_velocity**2 + v_velocity**2) ** 0.5
    direction = np.arctan2(u_velocity, v_velocity) * (180 / np.pi)

    # subtract the orientation angle of transect from compass North
    # then u is across channel
    if to_north:
        new_direction = direction + angle
    else:
        new_direction = direction - angle

    u_rotate = speed * np.sin(new_direction * (np.pi / 180))
    v_rotate = speed * np.cos(new_direction * (np.pi / 180))
    # u_rotate = speed * np.sin(angle * (np.pi / 180))
    # v_rotate = speed * np.cos(angle * (np.pi / 180))

    return u_rotate, v_rotate


def grid_angle(lon, lat):
    """Get angle using a metre grid transform. The angle may be off a bit if
    the grid cells do not have right angled corners.

    Args:
        lon (array): longitude of the grid
        lat (array): latitude of the grid

    Returns:
        array: the angle in degrees of the j grid lines relative to geographic
        North (i.e. clockwise from 12)
    """
    crs_wgs84 = pyproj.CRS("epsg:4326")
    angle = np.zeros(lon.shape)
    for j in tqdm(range(lon.shape[0] - 1)):
        for i in range(lon.shape[1] - 1):
            crs_aeqd = make_projection(lon[j, i], lat[j, i])
            transformer = pyproj.Transformer.from_crs(crs_wgs84, crs_aeqd)
            x_grid, y_grid = transformer.transform(lat[j + 1, i], lon[j + 1, i])
            angle[j, i] = np.arctan2(x_grid, y_grid)

    # differentiate to get the angle so copy last row one further and average
    angle[:, -1] = angle[:, -2]
    angle[-1, :] = angle[-2, :]

    # average angles in centre of array using cartesian coordinates
    xa = np.sin(angle)
    ya = np.cos(angle)
    xa[:, 1:-1] = (xa[:, :-2] + xa[:, 1:-1]) / 2
    ya[:, 1:-1] = (ya[:, :-2] + ya[:, 1:-1]) / 2
    xa[1:-1, :] = (xa[:-2, :] + xa[1:-1, :]) / 2
    ya[1:-1, :] = (ya[:-2, :] + ya[1:-1, :]) / 2
    angle = np.arctan2(xa, ya) * (180 / np.pi)
    return angle


def velocity_on_t(u_velocity, v_velocity):
    """Co-locate u and v onto the NEMO t-grid. Function from PyNEMO.

    Args:
        u_velocity (array): x-direction velocities along grid lines
        v_velocity (array): y-direction velocities along grid lines

    Returns:
        array, array: u and v velocity components co-located on the t-grid.
    """
    u_on_t_points = u_velocity * 1.0
    v_on_t_points = v_velocity * 1.0
    u_on_t_points[:, 1:] = 0.5 * (u_velocity[:, 1:] + u_velocity[:, :-1])
    v_on_t_points[1:, :] = 0.5 * (v_velocity[1:, :] + v_velocity[:-1, :])
    return u_on_t_points, v_on_t_points


def velocity_grid_to_geo(lon, lat, uv_velocity, polar_stereo_cartopy_bug_fix=False):
    """Makes combined adjustments to gridded velocities to make them plot
    with intuitive direction as quivers or streamlines in maps.
    (Developed and tested with the NEMO tripolar "ORCA" grid.)

    Args:
        lon (array): longitude of the grid
        lat (array): latitude of the grid
        uv_velocity (array): u and v velocities along grid lines
        polar_stereo_cartopy_bug_fix (bool, optional): Addesses a plotting bug in CartoPy=0.21.1
                            If True, makes an additional adjustment to the velocity
                            for plotting them on a stereographic (NorthPolarStereo or
                            SouthPolarStereo) projection in CartoPy. Defaults to False.


    Returns:
        array, array: NEMO grid u and v velocities that have been aligned
        in the north and east directions for plotting on the t-grid
    """

    u_on_t, v_on_t = velocity_on_t(uv_velocity[0], uv_velocity[1])
    angle_to_north = grid_angle(lon, lat)
    u_new, v_new = velocity_rotate(u_on_t, v_on_t, angle_to_north)
    if polar_stereo_cartopy_bug_fix:
        u_new, v_new = velocity_polar_bug_fix(u_new, v_new, lat)

    return u_new, v_new


def plot_polar_contour(lon, lat, var, ax_in, **kwargs):
    """
    Interpolate the data onto a regular grid with no north fold
    Generate new grid on NSIDC Polar North Stereographic projection on WGS84

    Args:
        lon (array): longitude coordinate of the variable
        lat (array): latitude coordinate of the variable
        var (array): variable to plot
        ax_in (axis): axis to plot contours on
        **kwargs (dict): variables for plotting a contour

    Returns:
        plot object: can be used for making a colorbar
    """
    try:
        import cartopy.crs as ccrs  # mapping plots
    except ImportError:
        warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        sys.exit(-1)

    crs_ps = ccrs.CRS("epsg:3413")  # North pole projection
    crs_wgs84 = ccrs.CRS("epsg:4326")
    # NSIDC grid
    x_grid, y_grid = np.meshgrid(np.linspace(-3850, 3750, 304) * 1000, np.linspace(-5350, 5850, 448) * 1000)
    grid = crs_wgs84.transform_points(crs_ps, x_grid, y_grid)
    # output is x, y, z triple but we don't need z
    lon_grid = grid[:, :, 0]
    lat_grid = grid[:, :, 1]
    points = np.vstack((lon.flatten(), lat.flatten())).T
    grid_var = si.griddata(points, var.flatten(), (lon_grid, lat_grid), method="linear")
    cs_out = ax_in.contour(x_grid, y_grid, grid_var, transform=ccrs.epsg(3413), **kwargs)
    return cs_out


def set_circle(ax_in):
    """
    Compute a circle in axes coordinates, which we can use as a boundary
    for the map. We can pan/zoom as much as we like - the boundary will be
    permanently circular.

    Args:
        ax_in (axis): axis object
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax_in.set_boundary(circle, transform=ax_in.transAxes)
