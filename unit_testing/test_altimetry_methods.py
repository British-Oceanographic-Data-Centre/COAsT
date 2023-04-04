"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast._utils import general_utils
import unittest
import numpy as np
import os.path as path
import xarray as xr
import matplotlib.pyplot as plt
import unit_test_files as files


class test_altimetry_methods(unittest.TestCase):
    def test_altimetry_load_subset_and_comparison(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)

        with self.subTest("Load example altimetry file"):
            altimetry = coast.Altimetry(files.fn_altimetry, config=files.fn_altimetry_config)

            # Test the data has loaded using attribute comparison, as for nemo_data
            alt_attrs_ref = dict(
                [
                    ("source", "Jason-1 measurements"),
                    ("date_created", "2019-02-20T11:20:56Z"),
                    ("institution", "CLS, CNES"),
                    ("Conventions", "CF-1.6"),
                ]
            )

            # checking is LHS is a subset of RHS
            check1 = alt_attrs_ref.items() <= altimetry.dataset.attrs.items()
            self.assertTrue(check1, "check1")

        with self.subTest("Subset altimetry data"):
            ind = altimetry.subset_indices_lonlat_box([-10, 10], [45, 60])
            ind = ind[::4]
            altimetry_nwes = altimetry.isel(t_dim=ind)  # nwes = northwest europe shelf

            check1 = altimetry_nwes.dataset.dims["t_dim"] == 54
            self.assertTrue(check1, "check1")

        with self.subTest("Interpolate Model to altimetry"):
            altimetry_nwes.obs_operator(sci, "ssh")
            # Check new variable is in altimetry dataset and isn't all NaNs
            check1 = False in np.isnan(altimetry_nwes.dataset.interp_ssh)
            self.assertTrue(check1, "check1")

        with self.subTest("Calculate CRPS for altimetry"):
            crps = altimetry_nwes.crps(sci, "ssh", "ocean_tide_standard_name")

            # TEST: Check length of crps and that it contains values
            check1 = crps.dataset.crps.shape[0] == altimetry_nwes.dataset.ocean_tide_standard_name.shape[0]
            check2 = False in np.isnan(crps.dataset.crps)
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Altimetry stats methods"):
            stats = altimetry_nwes.basic_stats("ocean_tide_standard_name", "interp_ssh")
            altimetry_nwes.basic_stats("ocean_tide_standard_name", "interp_ssh", create_new_object=False)

            # TEST: Check new object resembles internal object
            check1 = all(stats.dataset.error == altimetry_nwes.dataset.error)
            # TEST: Check lengths and values
            check2 = stats.dataset.absolute_error.shape[0] == altimetry_nwes.dataset.ocean_tide_standard_name.shape[0]
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Altimetry quick plot"):
            fig, ax = crps.quick_plot("crps")
            fig.savefig(files.dn_fig + "altimetry_crps_quick_plot.png")
            plt.close("all")
