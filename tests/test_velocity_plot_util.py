""" tests for plotting preparation methods.
 At the time of writing specifically targeting cartopy vector plots of polar domains """
# IMPORT modules. Must have pytest.

# import os.path as path
# import pytest
import pyproj
import numpy as np
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt

# import sys
from coast._utils import plot_util

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
# import unit_test_files as files


# Define a test function. Absolutely fine to have one or multiple per file.


def test_velocity_polar_bug_fix():
    """Test the plot_util.velocity_polar_bug_fix function."""
    lat = np.array([45, 55, 65, 75, 85])
    u_velocity = np.array([1, 1, 1, 1, 1])
    v_velocity = np.array([1, 1, 1, 1, 1])
    u_new, v_new = plot_util.velocity_polar_bug_fix(u_velocity, v_velocity, lat)

    result1 = np.array([1.15470054, 1.22674459, 1.30265868, 1.36910064, 1.4088727])
    assert np.isclose(u_new, result1).all()
    result2 = np.array([0.81649658, 0.70363179, 0.55052735, 0.35434932, 0.12279135])
    assert np.isclose(v_new, result2).all()


def test_make_projection():
    """Test the plot_util.make_projection function."""
    x_origin = 5  # East
    y_origin = 50  # North
    test_proj = plot_util.make_projection(x_origin, y_origin)
    assert test_proj.prime_meridian.unit_name == "degree"
    assert isinstance(test_proj, pyproj.CRS)


def test_velocity_rotate():
    """Test the plot_util.velocity_rotate function."""
    u_velocity = 1
    v_velocity = 0
    angle = 20

    u_rotate, v_rotate = plot_util.velocity_rotate(u_velocity, v_velocity, angle)
    assert np.isclose(u_rotate, 0.93969) & np.isclose(v_rotate, -0.34202)
    u_rotate, v_rotate = plot_util.velocity_rotate(u_velocity, v_velocity, angle, to_north=False)
    assert np.isclose(u_rotate, 0.93969) & np.isclose(v_rotate, 0.34202)


def test_grid_angle():
    """Test the plot_util.grid_angle function."""
    lat = np.array(([50, 51], [50, 51]))
    lon = np.array(([5, 5], [6, 6]))
    angle = plot_util.grid_angle(lon, lat)
    assert np.isclose(angle, 90, rtol=0.5).all()
    angle = plot_util.grid_angle(lat, lon)
    assert np.isclose(angle, 0, rtol=0.5).all()


def test_velocity_on_t():
    """Test the plot_util.velocity_on_t function."""
    u_velocity = np.array(([1, 2, 2], [2, 2, 1], [3, 3, 3]))
    v_velocity = np.array(([3, 4, 4], [4, 4, 3], [1, 1, 1]))
    u_on_t_points, v_on_t_points = plot_util.velocity_on_t(u_velocity, v_velocity)

    result1 = np.array(([1, 1.5, 2], [2, 2, 1.5], [3, 3, 3]))
    assert np.isclose(u_on_t_points, result1).all()
    result2 = np.array(([3, 4, 4], [3.5, 4, 3.5], [2.5, 2.5, 2]))
    assert np.isclose(v_on_t_points, result2).all()


def test_velocity_grid_to_geo():
    """Test the plot_util.velocity_grid_to_geo function."""
    lat = np.array(([50, 48, 46], [60, 58, 56], [70, 68, 66]))  # y, x
    lon = np.array(([5, 8, 11], [6, 9, 12], [7, 10, 13]))
    u_velocity = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))
    v_velocity = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))
    uv_velocity = [u_velocity, v_velocity]

    u_new, v_new = plot_util.velocity_grid_to_geo(lon, lat, uv_velocity, polar_stereo_cartopy_bug_fix=False)
    u_result1 = np.array(
        (
            [1.04903051, 1.05046004, 1.05188719],
            [1.04144945, 1.04295465, 1.04445727],
            [1.03380243, 1.03538430, 1.03696338],
        )
    )
    v_result1 = np.array(
        (
            [0.94843818, 0.94685463, 0.94526893],
            [0.95675652, 0.95511549, 0.95347209],
            [0.96501427, 0.96331685, 0.96161685],
        )
    )
    assert np.isclose(u_new, u_result1).all() & np.isclose(v_new, v_result1).all()

    u_new, v_new = plot_util.velocity_grid_to_geo(lon, lat, uv_velocity, polar_stereo_cartopy_bug_fix=True)
    print(u_new, v_new)
    u_result2 = np.array(
        ([1.222728, 1.21099989, 1.19965573], [1.28512188, 1.27230937, 1.25958666], [1.34721927, 1.33542729, 1.32321739])
    )
    v_result2 = np.array(
        (
            [0.71058865, 0.73039665, 0.74888326],
            [0.59030649, 0.61743735, 0.64299413],
            [0.43011655, 0.46543951, 0.49909492],
        )
    )
    assert np.isclose(u_new, u_result2).all() & np.isclose(v_new, v_result2).all()


def test_plot_polar_contour():
    """Test the plot_util.plot_polar_contour function."""
    lat = np.array(([50, 48, 46], [60, 58, 56], [70, 68, 66]))  # y, x
    lon = np.array(([5, 8, 11], [6, 9, 12], [7, 10, 13]))
    temp = np.array(([2, 1, 0], [2, 1, 0], [2, 2, 1]))
    figsize = (5, 5)  # Figure size
    mrc = ccrs.NorthPolarStereo(central_longitude=0.0)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.75], projection=mrc)
    cs1 = plot_util.plot_polar_contour(lon, lat, temp, ax1)
    assert isinstance(cs1, cartopy.mpl.contour.GeoContourSet)


def test_set_circle():
    """Test the plot_util.set_circle function."""
    figsize = (5, 5)  # Figure size
    mrc = ccrs.NorthPolarStereo(central_longitude=0.0)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.75], projection=mrc)
    try:
        plot_util.set_circle(ax1)
        assert True
    except AssertionError:
        assert False
