"""
TEST reading of WOD profiles
"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import os.path as path

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files

# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_WOD_read_data(unittest.TestCase):
    def test_load_WOD(self):
        
        with self.subTest("Load profile data from WOD"):
            
            WOD_profile_1D = coast.Profile(config=fn_WOD_config)
            WOD_profile_1D.read_WOD(fn_WOD)
            
            check1 = type(WOD_profile_1D) == coast.Profile
            self.assertTrue(check1, "check1")
            
    def test_reshape_WOD(self):            
        WOD_profile_1D = coast.Profile(config=fn_WOD_config)
        WOD_profile_1D.read_WOD(fn_WOD)            
        
        with self.subTest('Check reshape'):
            WOD_profile = coast.Profile.reshape_2D(WOD_profile_1D, 'DIC')
            
            check1 = type(WOD_profile) == coast.profile.Profile
            check2 = list(WOD_profile.dataset.coords) == ["latitude", "longitude", "time"]
            
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

