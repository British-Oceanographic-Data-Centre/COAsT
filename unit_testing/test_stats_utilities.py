'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import unittest
import coast
from coast import stats_util
import numpy as np
import unit_test_files as files
import datetime

class test_stats_utilities(unittest.TestCase):
    
    def test_find_maxima(self):
        date0 = datetime.datetime(2007, 1, 15)
        date1 = datetime.datetime(2007, 1, 16)
        tg = coast.Tidegauge(files.fn_tidegauge, date_start=date0, date_end=date1)
    
        tt, hh = stats_util.find_maxima(tg.dataset.time, tg.dataset.sea_level, method="comp")
        check1 = np.isclose((tt.values[0] - np.datetime64("2007-01-15T00:15:00")) / np.timedelta64(1, "s"), 0)
        check2 = np.isclose(hh.values[0], 1.027)
    
        tt, hh = stats_util.find_maxima(tg.dataset.time, tg.dataset.sea_level, method="cubic")
        check3 = np.isclose((tt[0] - np.datetime64("2007-01-15T00:07:49")) / np.timedelta64(1, "s"), 0)
        check4 = np.isclose(hh[0], 1.0347638302097757)
        
        self.assertTrue(check1, 'check1')
        self.assertTrue(check2, 'check2')
        self.assertTrue(check3, 'check3')
        self.assertTrue(check4, 'check4')
