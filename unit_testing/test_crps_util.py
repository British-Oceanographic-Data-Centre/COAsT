"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime
from coast import crps_util as cu

class test_crps_util(unittest.TestCase):
    def test_crps_empirical_2d(self):
        sample = np.array([[3,4,5,5],[6,7,8,2]])
        obs = 5

        crps = cu.crps_empirical(sample, obs)
        #print(f"crps: {crps}")

        # Check CRPS is as expected
        check1 = np.isclose(crps, 0.15625, rtol=0.0001)

        self.assertTrue(check1, "check1")

    def test_crps_empirical_nan(self):
        sample = np.array([np.nan,5,5,5])
        obs = 5

        crps = cu.crps_empirical(sample, obs)
        #print(f"crps: {crps}")

        # Check CRPS is as expected
        check1 = np.isclose(crps, 0.0, rtol=0.0001)

        self.assertTrue(check1, "check1")
