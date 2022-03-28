"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast import general_utils
import unittest
import numpy as np
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
