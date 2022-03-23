'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import os.path as path
import xarray as xr

# FILE NAMES to use for this testing module
dn_files = "../example_files/"
fn_nemo_harmonics = "coast_nemo_harmonics.nc"
fn_nemo_harmonics_dom = "coast_nemo_harmonics_dom.nc"

dn_config = "../config"
fn_config_t_grid = path.join(dn_config, "example_nemo_grid_t.json")
fn_config_f_grid = path.join(dn_config, "example_nemo_grid_f.json")
fn_config_u_grid = path.join(dn_config, "example_nemo_grid_u.json")
fn_config_v_grid = path.join(dn_config, "example_nemo_grid_v.json")
fn_config_w_grid = path.join(dn_config, "example_nemo_grid_w.json")

class test_gridded_harmonics(unittest.TestCase):
    
    def test_combine_and_convert_harmonics(self):
        
        harmonics = coast.Gridded(dn_files + fn_nemo_harmonics, 
                                  dn_files + fn_nemo_harmonics_dom, 
                                  config=fn_config_t_grid)
        
        with self.subTest("test_combine_harmonics"):
            constituents = ["K1", "M2", "S2", "K2"]
            harmonics_combined = harmonics.harmonics_combine(constituents)
    
            # TEST: Check values in arrays and constituents
            check1 = list(harmonics_combined.dataset.constituent.values) == constituents
            check2 = np.array_equal(harmonics_combined.dataset.harmonic_x[1].values, 
                                    harmonics.dataset.M2x.values)
            
            self.assertTrue(check1, msg='check1')
            self.assertTrue(check2, msg='check2')
        
        with self.subTest("test_convert_harmonics"):
        
            harmonics_combined.harmonics_convert(direction="cart2polar")
            harmonics_combined.harmonics_convert(direction="polar2cart", 
                                                 x_var="x_test", y_var="y_test")
    
            # TEST: Check variables and differences
            check1 = "x_test" in harmonics_combined.dataset.keys()
            diff = harmonics_combined.dataset.harmonic_x[0].values - harmonics_combined.dataset.x_test[0].values
            check2 = np.max(np.abs(diff)) < 1e-6
            
            self.assertTrue(check1, msg='check1')
            self.assertTrue(check2, msg='check2')
    

