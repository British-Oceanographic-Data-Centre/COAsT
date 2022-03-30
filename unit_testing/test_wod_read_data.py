"""
TEST reading of wod profiles
"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import unit_test_files as files
import numpy as np
import os.path as path

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files

# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_wod_read_data(unittest.TestCase):
    def test_load_wod(self):

        with self.subTest("Load profile data from WOD"):

            wod_profile_1D = coast.Profile(config=files.fn_wod_config)
            wod_profile_1D.read_wod(files.fn_wod)

            check1 = type(wod_profile_1D) == coast.Profile
            self.assertTrue(check1, "check1")

    def test_reshape_wod(self):
        wod_profile_1D = coast.Profile(config=files.fn_wod_config)
        wod_profile_1D.read_wod(files.fn_wod)

        with self.subTest("Check reshape"):
            my_list = ["DIC", "Temperature", "Alkalinity"]
            wod_profile = coast.Profile.reshape_2d(wod_profile_1D, my_list)

            check1 = type(wod_profile) == coast.profile.Profile
            check2 = list(wod_profile.dataset.coords) == ["time", "latitude", "longitude"]

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")
