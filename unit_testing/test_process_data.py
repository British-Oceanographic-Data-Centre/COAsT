# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import xarray as xr
import unit_test_files as files


class test_process_data_methods(unittest.TestCase):
    def test_statsmodel_seasonal_decompose(self):
        gd_t = coast.Gridded(files.fn_nemo_grid_t_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        fake_time = np.arange(
            np.datetime64("2010-01-01"), np.datetime64("2014-01-01"), np.timedelta64(1, "M"), dtype="datetime64[M]"
        ).astype("datetime64[s]")
        fake_temp = (
            (np.arange(0, 48) * 0.05)[:, np.newaxis, np.newaxis, np.newaxis]
            + np.random.normal(0, 0.1, 48)[:, np.newaxis, np.newaxis, np.newaxis]
            + np.tile(gd_t.dataset.temperature[:-1, :2, :, :], (8, 1, 1, 1))
        )
        fake_data_array = xr.DataArray(
            fake_temp,
            coords={
                "t_dim": fake_time,
                "depth_0": gd_t.dataset.depth_0[:2, :, :],
                "longitude": gd_t.dataset.longitude,
                "latitude": gd_t.dataset.latitude,
            },
            dims=["t_dim", "z_dim", "y_dim", "x_dim"],
        )

        # Coast version
        proc_data = coast.ProcessData()
        grd = proc_data.seasonal_decomposition(fake_data_array, 4, model="additive", period=6, extrapolate_trend="freq")
        # statsmodel version
        from statsmodels.tsa.seasonal import seasonal_decompose

        stm_result = seasonal_decompose(
            fake_data_array.fillna(0).values.reshape((48, 2 * 375 * 297)),
            model="additive",
            period=6,
            extrapolate_trend="freq",
        )

        check_trend = np.allclose(stm_result.trend.reshape((48, 2, 375, 297)), grd.dataset.trend.compute().fillna(0))
        check_seasonal = np.allclose(
            stm_result.seasonal.reshape((48, 2, 375, 297)), grd.dataset.seasonal.compute().fillna(0)
        )
        check_residual = np.allclose(
            stm_result.resid.reshape((48, 2, 375, 297)), grd.dataset.residual.compute().fillna(0)
        )

        self.assertTrue(check_trend, "trends don't match")
        self.assertTrue(check_seasonal, "seasonals don't match")
        self.assertTrue(check_residual, "residuals don't match")
