import coast
import unittest
import numpy as np
import os.path as path

# File names to use for this testing module
dn_files = "./example_files/"
dn_config = "./config"
fn_nemo_dat = path.join(dn_files, "coast_example_nemo_data.nc")
fn_nemo_dom = path.join(dn_files, "coast_example_nemo_domain.nc")
fn_config_t_grid = path.join(dn_config, "example_nemo_grid_t.json")

# Single unit test. Can contain multiple test methods and subTests.
class test_xesmf_convert(unittest.TestCase):

    # Test for conversion from gridded to xesmf.
    # Here I've used one test and then subtests for each smaller test.
    # This could also be split into multiple methods but the file would need
    # to be loaded multiple times. Using subtests allows a sequential testing.
    def test_basic_conversion_to_xesmf(self):

        # Read data files
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)

        # Convert for single file
        with self.subTest("xesmf_convert for single gridded obj"):
            xesmf_ready = coast.xesmf_convert(sci)
            check_grid = xesmf_ready.input_grid
            check_data = xesmf_ready.input_data
            check1 = np.array_equal(check_grid.lat.values, sci.dataset.latitude.values)
            check2 = np.array_equal(
                check_data.temperature[0, 0].values, sci.dataset.temperature[0, 0].values, equal_nan=True
            )
            self.assertTrue(check1, "Test")
            self.assertTrue(check2, "Test")

        # Convert for two files
        with self.subTest("xesmf_convert for two gridded obj"):
            xesmf_ready = coast.xesmf_convert(sci, sci)
            check_grid = xesmf_ready.input_grid
            check_data = xesmf_ready.input_data
            check1 = np.array_equal(check_grid.lat.values, sci.dataset.latitude.values)
            check2 = np.array_equal(
                check_data.temperature[0, 0].values, sci.dataset.temperature[0, 0].values, equal_nan=True
            )
            self.assertTrue(check1, "Test")
            self.assertTrue(check2, "Test")

    # OR putting subtests in their own routines::
    def test_xesmf_convert_for_one_object(self):

        # Read data files
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)

        # Convert for single file
        xesmf_ready = coast.xesmf_convert(sci)
        check_grid = xesmf_ready.input_grid
        check_data = xesmf_ready.input_data
        check1 = np.array_equal(check_grid.lat.values, sci.dataset.latitude.values)
        check2 = np.array_equal(
            check_data.temperature[0, 0].values, sci.dataset.temperature[0, 0].values, equal_nan=True
        )
        check1 = False
        self.assertTrue(check1, "Test")
        self.assertTrue(check2, "Test")

    def test_xesmf_convert_for_two_objects(self):

        # Read data files
        sci = coast.Gridded(fn_nemo_dat, fn_nemo_dom, config=fn_config_t_grid)

        xesmf_ready = coast.xesmf_convert(sci, sci)
        check_grid = xesmf_ready.input_grid
        check_data = xesmf_ready.input_data
        check1 = np.array_equal(check_grid.lat.values, sci.dataset.latitude.values)
        check2 = np.array_equal(
            check_data.temperature[0, 0].values, sci.dataset.temperature[0, 0].values, equal_nan=True
        )
        self.assertTrue(check1, "Test")
        self.assertTrue(check2, "Test")
