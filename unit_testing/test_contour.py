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
class test_contour_methods(unittest.TestCase):
    
    def setUp(self):
        # This is called at the beginning of every test_ method. Optional.
        return
    
    def tearDown(self):
        # This is called at the end of every test_ method. Optional.
        return
    
    @classmethod
    def setUpClass(cls):
        # This is called at the beginning of every test CLASS. Optional
        return
    
    @classmethod
    def tearDownClass(cls):
        # This is called at the end of every test CLASS. Optional.
        return
    
    # TEST METHODS  
    def test_along_contour_flow(self):
        # load gridded data
        nemo_t = coast.Gridded(fn_domain = fn_nemo_dom, config = fn_config_t_grid)
        nemo_u = coast.Gridded(fn_nemo_grid_u_dat, fn_nemo_dom, config = fn_config_u_grid)
        nemo_v = coast.Gridded(fn_nemo_grid_v_dat, fn_nemo_dom, config = fn_config_v_grid)
        # create contour dataset along 200 m isobath
        contours, no_contours = coast.Contour.get_contours(nemo_t, 200)
        y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_t, contours[0], [50, -10], [60, 3])
        cont_t = coast.ContourT(nemo_t, y_ind, x_ind, 200)
        # calculate flow along contour
        cont_t.calc_along_contour_flow(nemo_u, nemo_v)
        
        if np.allclose(
            (cont_f.data_cross_flow.normal_velocities + cont_f.data_cross_flow.depth_integrated_normal_transport).sum(),
            -1152.3771,
        ):
            print(str(sec) + chr(subsec) + " OK - Cross-contour flow calculations as expected")
        else:
            print(str(sec) + chr(subsec) + " X - Cross-contour flow calculations not as expected")
    

    def test_successful_method(self):
        # Test things. Use assert methods from unittest. Some true tests:
        self.assertTrue(1==1, "Will not see this.")
        self.assertEqual(1,1, "Will not see this.")
        self.assertIsNone(None, "Will not see this.")
        
        
    def test_bad_method(self):
       # Test things. Use assert methods from unittest. This will fail
        self.assertTrue(1==2, "1 does not equal 2")
        self.assertEqual(1,2, "1 does not equal 2.")
        self.assertIsNone(1, "1 does not none.")
        
    def test_with_subtests(self):
        # Sometimes you might want to do a test within a parameter loop or
        # a bunch of sequential tests (so you don't have to read data 
        # repeatedly). Subtest can be used for this as follows:
            
        with self.subTest("First subtest."):
            a = 1
            self.assertEqual(a,1,"Will not see this.")
            
        with self.subTest("Second subtest"):
            b = 1
            self.assertEqual(a,b, "will not see this")
            
        with self.subTest("Third subtest"):
            c =  50000000
            self.assertAlmostEqual(a, c, msg="0 is not almost equal to 50000000")