'''
TEMPLATE testing file, usin unittest package.
Please save this file with a name starting with "test_".
TestCase classes have a whole bunch of methods available to them. Some of them
are showcased below. You can also add your own methods to them. Anything you
want tested by the unit testing system should start with "test_".
For more info on assert test cases, see:
    https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertTrue
'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import os.path as path

# FILE NAMES to use for this testing module
dn_files = "./example_files"
dn_config = "./config"
fn_nemo_dom = path.join(dn_files, "coast_example_nemo_domain.nc")
fn_nemo_grid_u_dat = path.join(dn_files, "nemo_data_U_grid.nc")
fn_nemo_grid_v_dat = path.join(dn_files, "nemo_data_V_grid.nc")
fn_config_f_grid = path.join(dn_config, "example_nemo_grid_f.json")
fn_config_t_grid = path.join(dn_config, "example_nemo_grid_t.json")
fn_config_u_grid = path.join(dn_config, "example_nemo_grid_u.json")
fn_config_v_grid = path.join(dn_config, "example_nemo_grid_v.json")

# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_ContourT_methods(unittest.TestCase):
    
    def setUp(self):
        # This is called at the beginning of every test_ method.
        # load gridded data
        self.nemo_t = coast.Gridded(fn_domain = fn_nemo_dom, config = fn_config_t_grid)
        self.nemo_u = coast.Gridded(fn_nemo_grid_u_dat, fn_nemo_dom, config = fn_config_u_grid)
        self.nemo_v = coast.Gridded(fn_nemo_grid_v_dat, fn_nemo_dom, config = fn_config_v_grid)
        # create contour dataset along 200 m isobath
        contours, no_contours = coast.Contour.get_contours(self.nemo_t, 200)
        y_ind, x_ind, contour = coast.Contour.get_contour_segment(self.nemo_t, contours[0], [50, -10], [60, 3])
        self.cont_t = coast.ContourT(self.nemo_t, y_ind, x_ind, 200)
           
    # TEST METHODS  
    def test_along_contour_flow(self):
        # calculate flow along contour
        self.cont_t.calc_along_contour_flow(self.nemo_u, self.nemo_v)
        
        with self.subTest("Check on velocities"):
            cksum = (self.cont_t.data_along_flow.velocities 
                 * self.cont_t.data_along_flow.e3 
                 * self.cont_t.data_along_flow.e4).sum().values
            self.assertTrue(np.isclose(cksum, 116660850), "velocities checksum: " + str(cksum) + ", should be 116660850" )

        with self.subTest("Check on transport"):
            cksum = (self.cont_t.data_along_flow.transport  
                 * self.cont_t.data_along_flow.e4).sum().values
            self.assertTrue(np.isclose(cksum, 116660850), "transports checksum: " + str(cksum) + ", should be 116660850" )
    
    def test_along_contour_2d_flow(self):
        # calculate flow along contour
        self.nemo_u.dataset = self.nemo_u.dataset.isel(z_dim = 0).squeeze()
        self.nemo_v.dataset = self.nemo_v.dataset.isel(z_dim = 0).squeeze()
        self.cont_t.calc_along_contour_flow_2d(self.nemo_u, self.nemo_v)
        
        cksum = (self.cont_t.data_along_flow.velocities 
                 * self.cont_t.data_along_flow.e3_0 
                 * self.cont_t.data_along_flow.e4).sum().values
        self.assertTrue(np.isclose(cksum, 293910.94), "velocities checksum: " + str(cksum) + ", should be 293910.94" )

    