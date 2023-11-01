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
from coast._utils import plot_util
import unittest
import numpy as np
import os.path as path
import pyproj
from shapely.geometry import box, Point

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files


# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_velocity_plot_util(unittest.TestCase):

    # TEST METHODS
    def test_make_projection(self):
        x_origin = 5 # East
        y_origin = 50 # North
        test_proj = plot_util.make_projection(x_origin, y_origin)
        bounding_box = box(test_proj.area_of_use.bounds)
        check1 = type(test_proj) == pyproj.crs.CRS
        check2 = Point(x_origin, y_origin).within(bounding_box)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_rotate(self):
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
        lat = np.array(([50, 51], [50, 51]))
        lon = np.array(([5, 5], [6, 6]))
        angle = plot_util.grid_angle(lon, lat)
        check1 = np.isclose(angle, 0)
        angle = plot_util.grid_angle(lat, lon)
        check2 = np.isclose(angle, 90)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_on_t(self):
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

"""
    def test_velocity_polar(self):
        lat = np.array([45, 55, 65, 75, 85])
        u_velocity = np.array([1, 1, 1, 1, 1])
        v_velocity = np.array([1, 1, 1, 1, 1])
        u_new, v_new = plot_util.velocity_polar(u_velocity, v_velocity, lat)

        result1 = np.array([])
        check1 = np.isclose(u_new == result1).all()
        result2 = np.array([])
        check2 = np.isclose(v_new == result2).all()
        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_velocity_grid_to_geo(self):
        u_new, v_new = velocity_grid_to_geo(lon, lat, u_velocity, v_velocity, polar_stereo=False)
"""