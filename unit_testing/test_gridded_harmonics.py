"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import unit_test_files as files


class test_gridded_harmonics(unittest.TestCase):
    def test_combine_and_convert_harmonics(self):

        harmonics = coast.Gridded(files.fn_nemo_harmonics, files.fn_nemo_harmonics_dom, config=files.fn_config_t_grid)

        with self.subTest("test_combine_harmonics"):
            constituents = ["K1", "M2", "S2", "K2"]
            harmonics_combined = harmonics.harmonics_combine(constituents)

            # TEST: Check values in arrays and constituents
            check1 = list(harmonics_combined.dataset.constituent.values) == constituents
            check2 = np.array_equal(harmonics_combined.dataset.harmonic_x[1].values, harmonics.dataset.M2x.values)

            self.assertTrue(check1, msg="check1")
            self.assertTrue(check2, msg="check2")

        with self.subTest("test_convert_harmonics"):

            harmonics_combined.harmonics_convert(direction="cart2polar")
            harmonics_combined.harmonics_convert(direction="polar2cart", x_var="x_test", y_var="y_test")

            # TEST: Check variables and differences
            check1 = "x_test" in harmonics_combined.dataset.keys()
            diff = harmonics_combined.dataset.harmonic_x[0].values - harmonics_combined.dataset.x_test[0].values
            check2 = np.max(np.abs(diff)) < 1e-6

            self.assertTrue(check1, msg="check1")
            self.assertTrue(check2, msg="check2")
