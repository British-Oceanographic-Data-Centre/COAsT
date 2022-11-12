"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast._utils import general_utils
import unittest
import numpy as np
import xarray as xr
import pytz
import datetime
import unit_test_files as files


class test_general_utils(unittest.TestCase):
    def test_copy_coast_object(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        sci_copy = sci.copy()
        check1 = sci_copy.dataset == sci.dataset
        self.assertTrue(check1, msg="check1")

    def test_getitem(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        check1 = sci.dataset["ssh"].equals(sci["ssh"])
        self.assertTrue(check1, msg="check1")

    def test_coast_variable_renaming(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        sci_copy = sci.copy()
        sci_copy.rename({"ssh": "renamed"})
        check1 = sci["ssh"].equals(sci_copy["renamed"])
        self.assertTrue(check1, "check1")

    def test_day_of_week(self):
        check1 = general_utils.day_of_week(np.datetime64("2020-10-16")) == "Fri"
        self.assertTrue(check1, msg="check1")

    def test_bst_to_gmt(self):
        time_str = "11/10/2020 12:00"
        datetime_obj = datetime.datetime.strptime(time_str, "%d/%m/%Y %H:%M")
        bst_obj = pytz.timezone("Europe/London")
        check1 = np.datetime64(bst_obj.localize(datetime_obj).astimezone(pytz.utc)) == np.datetime64(
            "2020-10-11T11:00:00"
        )
        self.assertTrue(check1, msg="check1")

    def test_nan_helper(self):
        y = np.array([np.NaN, 1, 1, np.NaN, np.NaN, 7, 2, np.NaN, 0])
        y_xr = xr.DataArray(y)
        # numpy array
        nans, x = general_utils.nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        check1 = all(y == np.array([1.0, 1.0, 1.0, 3.0, 5.0, 7.0, 2.0, 1.0, 0.0]))
        # xarray
        nans, x = general_utils.nan_helper(y_xr)
        y_xr[nans] = np.interp(x(nans), x(~nans), y_xr[~nans])
        check2 = all(y_xr.values == np.array([1.0, 1.0, 1.0, 3.0, 5.0, 7.0, 2.0, 1.0, 0.0]))

        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")

    def test_fill_holes_1d(self):
        input = np.array([np.nan, np.nan, 2., np.nan, 4,5,6], dtype='float64')
        input_xr = xr.DataArray(input)
        target = np.array([2., 2., 2., 3., 4., 5., 6.])

        check1 = all(fill_holes_1d(input) == target)
        check2 = all(fill_holes_1d(input_xr).values == target)

        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")
