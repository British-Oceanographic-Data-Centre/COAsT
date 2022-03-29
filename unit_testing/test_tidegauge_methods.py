# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime


class test_tidegauge_methods(unittest.TestCase):
    def test_read_gesla_and_compare_to_model(self):

        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        sci.dataset["landmask"] = sci.dataset.bottom_level == 0

        with self.subTest("Read GESLA file"):
            date0 = datetime.datetime(2007, 1, 10)
            date1 = datetime.datetime(2007, 1, 12)
            lowestoft = coast.Tidegauge()
            lowestoft.read_gesla_v3(files.fn_tidegauge, date_start=date0, date_end=date1)

            # TEST: Define Attribute dictionary for comparison
            test_attrs = {
                "site_name": "Lowestoft",
                "country": "United_Kingdom",
                "contributor": "British_Oceanographic_Data_Centre",
                "coordinate_system": "UK",
                "original_start_date": np.datetime64("1964-01-01 00:00:00"),
                "original_end_date": np.datetime64("2014-12-31 23:45:00"),
                "time_zone_hours": 0.0,
                "precision": 0.001,
                "null_value": -99.9999,
            }

            # TEST: Check attribute dictionary and length of sea_level.
            check1 = len(lowestoft.dataset.ssh.isel(id=0)) == 193
            check2 = lowestoft.dataset.attrs == test_attrs
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Plot Single GESLA"):
            f, a = lowestoft.plot_on_map()
            f.savefig(files.dn_fig + "tidegauge_map.png")
            plt.close("all")

        with self.subTest("Obs operator"):
            lowestoft.obs_operator(sci, time_interp="linear")

            # TEST: Check that the resulting interp_sossheig variable is of the same
            # length as sea_level and that it is populated.
            interp = lowestoft.dataset.interp_ssh
            interp_len = interp.shape[0]
            orig_len = lowestoft.dataset.sea_level.shape[0]
            check1 = interp_len == orig_len
            check2 = False in np.isnan(lowestoft.dataset.interp_ssh)
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Tidegauge CRPS"):
            crps = lowestoft.crps(sci, "ssh")

            # TEST: Check length of crps and that it contains values
            check1 = crps.dataset.crps.shape[0] == lowestoft.dataset.sea_level.shape[0]
            check2 = False in np.isnan(crps.dataset.crps)
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Tidegauge stats"):
            stats = lowestoft.basic_stats("sea_level", "interp_ssh")
            lowestoft.basic_stats("sea_level", "interp_ssh", create_new_object=False)

            # TEST: Check new object resembles internal object
            check1 = all(stats.dataset.error == lowestoft.dataset.error)
            # TEST: Check lengths and values
            check2 = stats.dataset.absolute_error.shape[0] == lowestoft.dataset.sea_level.shape[0]
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

    def test_tidegauge_resample_and_apply_doodsonx0(self):

        with self.subTest("Resample to 1H"):
            date0 = datetime.datetime(2007, 1, 10)
            date1 = datetime.datetime(2007, 1, 12)
            lowestoft = coast.Tidegauge(files.fn_tidegauge, date_start=date0, date_end=date1)
            lowestoft.resample_mean("sea_level", "1H")
            td0 = lowestoft.dataset.time_1H[1] - lowestoft.dataset.time_1H[0]
            check1 = td0.values.astype("timedelta64[h]") == np.timedelta64(1, "h")
            # TEST: Check length
            check2 = np.ceil(lowestoft.dataset.time.shape[0] / 4) == lowestoft.dataset.time_1H.shape[0]
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Apply Doodson x0 filter"):
            lowestoft.apply_doodson_x0_filter("sea_level_1H")

            # TEST: Check new times are same length as variable
            check1 = lowestoft.dataset.time_1H.shape == lowestoft.dataset.sea_level_1H_dx0.shape
            # TEST: Check there are number values in output
            check2 = False in np.isnan(lowestoft.dataset.sea_level_1H_dx0)

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Plot time series"):
            f, a = lowestoft.plot_timeseries(["sea_level", "sea_level_1H", "sea_level_1H_dx0"])
            f.savefig(files.dn_fig + "tidegauge_timeseries.png")
            plt.close("all")

    def test_load_multiple_tidegauge(self):

        with self.subTest("Load multiple gauge"):
            date0 = datetime.datetime(2007, 1, 10)
            date1 = datetime.datetime(2007, 1, 12)
            lowestoft = coast.Tidegauge()
            lowestoft.read_gesla_v3(files.fn_tidegauge, date0, date1)
            multi_tg = coast.Tidegauge()
            tg_list = multi_tg.read_gesla_v3(files.fn_multiple_tidegauge, date_start=date0, date_end=date1)

            # TEST: Check length of list
            check1 = len(tg_list) == 2
            # TEST: Check lowestoft matches
            check2 = all(tg_list[1].dataset == lowestoft.dataset)

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Plot multiple gauge"):
            f, a = coast.Tidegauge.plot_on_map_multiple(tg_list)
            f.savefig(files.dn_fig + "tidegauge_multiple_map.png")
            plt.close("all")

    def test_tidegauge_for_tabulated_data(self):
        date_start = np.datetime64("2020-10-11 07:59")
        date_end = np.datetime64("2020-10-20 20:21")

        # Initiate a Tidegauge object, if a filename is passed it assumes it is a GESLA type object
        tg = coast.Tidegauge()
        tg.read_hlw(files.fn_gladstone, date_start, date_end)

        check1 = len(tg.dataset.ssh) == 37
        check2 = tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="nearest_HW").values == 8.01
        check3 = tg.get_tide_table_times(
            np.datetime64("2020-10-13 12:48"), method="nearest_1"
        ).time.values == np.datetime64("2020-10-13 14:36")
        check4 = np.array_equal(
            tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="nearest_2").values, [2.83, 8.01]
        )
        check5 = np.array_equal(
            tg.get_tide_table_times(np.datetime64("2020-10-13 12:48"), method="window", winsize=24).values,
            [3.47, 7.78, 2.8, 8.01, 2.83, 8.45, 2.08, 8.71],
        )

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
        self.assertTrue(check3, "check3")
        self.assertTrue(check4, "check4")
        self.assertTrue(check5, "check5")

    def test_tidegauge_finding_extrema(self):

        with self.subTest("Find extrema"):
            date0 = datetime.datetime(2007, 1, 10)
            date1 = datetime.datetime(2007, 1, 20)
            lowestoft2 = coast.Tidegauge()
            lowestoft2.read_gesla_v3(files.fn_tidegauge, date_start=date0, date_end=date1)

            # Use comparison of neighbourhood method (method="comp" is assumed)
            extrema_comp = lowestoft2.find_high_and_low_water("ssh", distance=40)
            # Check actual maximum/minimum is in output dataset
            check1 = np.nanmax(lowestoft2.dataset.ssh) in extrema_comp.dataset.ssh_highs
            check2 = np.nanmin(lowestoft2.dataset.ssh) in extrema_comp.dataset.ssh_lows
            # Check new time dimensions have correct length (hardcoded here)
            check3 = len(extrema_comp.dataset.time_highs) == 19
            check4 = len(extrema_comp.dataset.time_lows) == 18

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")
            self.assertTrue(check3, "check3")
            self.assertTrue(check4, "check4")

        with self.subTest("Plot extrema"):
            # Attempt a plot
            f = plt.figure()
            plt.plot(lowestoft2.dataset.time, lowestoft2.dataset.ssh[0])
            plt.scatter(extrema_comp.dataset.time_highs.values, extrema_comp.dataset.ssh_highs, marker="o", c="g")
            plt.scatter(extrema_comp.dataset.time_lows.values, extrema_comp.dataset.ssh_lows, marker="o", c="r")

            plt.legend(["Time Series", "Maxima", "Minima"])
            plt.title("Tide Gauge Optima at Lowestoft")
            f.savefig(files.dn_fig + "tidegauge_optima.png")
            plt.close("all")

    def test_tidegauge_cubic_spline_extrema(self):

        with self.subTest("Fit cubic spline"):
            date_start = np.datetime64("2020-10-12 23:59")
            date_end = np.datetime64("2020-10-14 00:01")

            # Initiate a Tidegauge object, if a filename is passed
            # it assumes it is a GESLA  type object
            tg = coast.Tidegauge()
            # specify the data read as a High Low Water dataset
            tg.read_bodc(files.fn_tidegauge2, date_start, date_end)

            # Use cubic spline fitting method
            extrema_cubc = tg.find_high_and_low_water("ssh", method="cubic")

            # Check actual maximum/minimum is in output dataset
            check1 = np.isclose(extrema_cubc.dataset.ssh_highs, [7.774, 7.91]).all()
            check2 = np.isclose(extrema_cubc.dataset.ssh_lows, [2.636, 2.547]).all()

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Attempt plot"):
            f = plt.figure()
            plt.plot(tg.dataset.time, tg.dataset.ssh[0])
            plt.scatter(extrema_cubc.dataset.time_highs.values, extrema_cubc.dataset.ssh_highs, marker="o", c="g")
            plt.scatter(extrema_cubc.dataset.time_lows.values, extrema_cubc.dataset.ssh_lows, marker="o", c="r")

            plt.legend(["Time Series", "Maxima", "Minima"])
            plt.title("Tide Gauge Optima at Gladstone, fitted cubic spline")
            f.savefig(files.dn_fig + "tidegauge_optima.png")
            plt.close("all")
