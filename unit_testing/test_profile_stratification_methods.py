"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime


class test_profile_stratification_methods(unittest.TestCase):
    def test_calculate_pea(self):
        profile = coast.Profile(config=files.fn_profile_config)
        profile.read_en4(files.fn_profile)
        profile.dataset = profile.dataset.isel(id_dim=np.arange(0, profile.dataset.dims["id_dim"], 10)).load()

        pa = coast.ProfileStratification(profile)
        Zmax = 200  # metres
        pa.calc_pea(profile, Zmax)

        check1 = np.isclose(pa.dataset.pea.mean(dim="id_dim").item(), 5.8750878507)
        self.assertTrue(check1, "check1")

        with self.subTest("Test quick_plot()"):
            fig, ax = pa.quick_plot("pea")
            fig.tight_layout()
            fig.savefig(files.dn_fig + "profile_pea.png")
            plt.close("all")
