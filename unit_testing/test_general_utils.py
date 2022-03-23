'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast import general_utils
import unittest
import numpy as np
import os.path as path
import xarray as xr

# FILE NAMES to use for this testing module
dn_files = "../example_files/"
fn_nemo_dat = path.join(dn_files, "coast_example_nemo_data.nc")
fn_nemo_dom = path.join(dn_files, "coast_example_nemo_domain.nc")

dn_config = "../config"
fn_config_t_grid = path.join(dn_config, "example_nemo_grid_t.json")

class test_general_utils(unittest.TestCase):
    
    def test_copy_coast_object(self):
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)
        sci_copy = sci.copy()
        check1 = sci_copy.dataset == sci.dataset
        self.assertTrue(check1, msg='check1')
        
    def test_getitem(self):
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)
        check1 = sci.dataset["ssh"].equals(sci["ssh"])
        self.assertTrue(check1, msg='check1')
        
    def test_coast_variable_renaming(self):
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)
        sci_copy = sci.copy()
        sci_copy.rename({"ssh": "renamed"})
        check1 = sci["ssh"].equals(sci_copy["renamed"])
        self.assertTrue(check1, 'check1')
        
    def test_day_of_week(self):
        check1 = general_utils.day_of_week(np.datetime64("2020-10-16")) == "Fri"
        self.assertTrue(check1, msg='check1')
    

