"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import unittest
import coast
import numpy as np
import unit_test_files as files
import os
import xarray as xr


class test_climatology(unittest.TestCase):
    def test_monthly_and_seasonal_climatology(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        ds = sci.dataset[["temperature", "ssh"]].isel(z_dim=0)
        clim = coast.Climatology()
        fn_out = os.path.join(files.dn_files, "test_climatology.nc")
        monthly = clim.make_climatology(ds, "month").load()
        seasonal = clim.make_climatology(ds, "season", fn_out=fn_out)

        # create dataset with missing values
        ds2 = ds.copy(deep=True)
        ds2["temperature"][::2, :100, :100] = np.nan
        ds2["ssh"][::2, :100, :100] = np.nan
        seaC = clim.make_climatology(ds2, "season")
        seaX = ds2.groupby("time.season").mean("t_dim")
        # throws error is not close
        xr.testing.assert_allclose(seaC, seaX)

        mn = mn = np.nanmean(ds.temperature, axis=0)
        check1 = np.nanmax(np.abs(mn - monthly.temperature[0])) < 1e-6
        check2 = os.path.isfile(fn_out)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
