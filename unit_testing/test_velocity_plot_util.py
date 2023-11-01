"""
TEMPLATE testing file, usin unittest package.
Please save this file with a name starting with "test_".
TestCase classes have a whole bunch of methods available to them. Some of them
are showcased below. You can also add your own methods to them. Anything you
want tested by the unit testing system should start with "test_".

For more info on assert test cases, see:
    https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertTrue
"""

# IMPORT modules. Must have unittest, and probably coast.
import os.path as path
import unittest
import numpy as np
import pyproj
from shapely.geometry import box, Point
from coast._utils import plot_util

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files


# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_velocity_plot_util(unittest.TestCase):
    """Class of methods for testing the velocity plotting functions in _utils/plot_util.py
    """

    def test_velocity_polar(self):
        """Test the plot_util.velocity_polar function.
        """
        lat = np.array([45, 55, 65, 75, 85])
        u_velocity = np.array([1, 1, 1, 1, 1])
        v_velocity = np.array([1, 1, 1, 1, 1])
        u_new, v_new = plot_util.velocity_polar(u_velocity, v_velocity, lat)

        result1 = np.array([1.15470054, 1.22674459, 1.30265868, 1.36910064, 1.4088727])
        check1 = np.isclose(u_new, result1).all()
        result2 = np.array([0.81649658, 0.70363179, 0.55052735, 0.35434932, 0.12279135])
        check2 = np.isclose(v_new, result2).all()
        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
    
    def test_make_projection(self):
        """Test the plot_util.make_projection function.
        """
        x_origin = 5 # East
        y_origin = 50 # North
        test_proj = plot_util.make_projection(x_origin, y_origin)
        bounding_box = box(test_proj.area_of_use.bounds)
        check1 = isinstance(test_proj, pyproj.crs.CRS)
        check2 = Point(x_origin, y_origin).within(bounding_box)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_rotate(self):
        """Test the plot_util.velocity_rotate function.
        """
        u_velocity = 1
        v_velocity = 0
        angle = 20

        u_rotate, v_rotate = plot_util.velocity_rotate(u_velocity, v_velocity, 
                                                       angle)
        check1 = np.isclose(u_rotate, 0.93969) & np.isclose(v_rotate, -0.34202)   
        u_rotate, v_rotate = plot_util.velocity_rotate(u_velocity, v_velocity, 
                                                       angle, to_north=False)
        check2 = np.isclose(u_rotate, 0.93969) & np.isclose(v_rotate, 0.34202)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_grid_angle(self):
        """Test the plot_util.grid_angle function.
        """
        lat = np.array(([50, 50], [51, 51]))
        lon = np.array(([5, 6], [5, 6]))
        angle = plot_util.grid_angle(lon, lat)
        check1 = np.isclose(angle, 0)
        angle = plot_util.grid_angle(lat, lon)
        check2 = np.isclose(angle, 90, rtol=0.5)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_on_t(self):
        """Test the plot_util.velocity_on_t function.
        """
        u_velocity = np.array(([1, 2, 2], [2, 2, 1], [3, 3, 3]))
        v_velocity = np.array(([3, 4, 4], [4, 4, 3], [1, 1, 1]))
        u_on_t_points, v_on_t_points = plot_util.velocity_on_t(
            u_velocity, v_velocity)
        
        result1 = np.array(([1, 1.5, 2], [2, 2, 1.5], [3, 3, 3]))
        check1 = np.isclose(u_on_t_points, result1).all()
        result2 = np.array(([3, 4, 4], [3.5, 4, 3.5], [2.5, 2.5, 2]))
        check2 = np.isclose(v_on_t_points, result2).all()
        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_grid_to_geo(self):
        """Test the plot_util.velocity_grid_to_geo function.
        """
        lat = np.array(([50, 60, 70], [48, 58, 68], [46, 56, 66]))
        lon = np.array(([5, 5, 5], [8, 8, 8], [11, 11, 11]))
        u_velocity = np.array(([1, 2, 2], [2, 2, 1], [3, 3, 3]))
        v_velocity = np.array(([0.5, 1.5, 1.5], [1.5, 1.5, 0.5], [1, 1, 1]))

        u_new, v_new = plot_util.velocity_grid_to_geo(lon, lat, u_velocity, v_velocity, polar_stereo=False)
        u_result1 = np.array([], [], [])
        v_result1 = np.array([], [], [])
        check1 = np.isclose(u_new, u_result1).all() & np.isclose(v_new, v_result1).all()

        u_new, v_new = plot_util.velocity_grid_to_geo(lon, lat, u_velocity, v_velocity, polar_stereo=True)
        u_result2 = np.array([], [], [])
        v_result2 = np.array([], [], [])
        check2 = np.isclose(u_new, u_result2).all() & np.isclose(v_new, v_result2).all()

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")