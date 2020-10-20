# Test with PyTest

import coast
import numpy as np
import xarray as xr
import datetime
import logging
import coast.general_utils as general_utils


#-----------------------------------------------------------------------------#
#%% ( 2d ) day of the week function                                           #
#                                                                             #
def test_dayoweek():
    check1 = general_utils.dayoweek( np.datetime64('2020-10-16') ) == 'Fri'
    assert check1

#-----------------------------------------------------------------------------#
#%% ( 7k ) TIDEGAUGE method for tabulated data                                #
#                                                                             #
if(0):
    def test_tidetable():

        filnam = 'example_files/Gladstone_2020-10_HLW.txt'
        date_start = np.datetime64('2020-10-11 07:59')
        date_end = np.datetime64('2020-10-20 20:21')

        # Initiate a TIDEGAUGE object, if a filename is passed it assumes it is a GESLA type object
        tg = coast.TIDEGAUGE()
        tg.dataset = tg.read_HLW_to_xarray(filnam, date_start, date_end)

        check1 = len(tg.dataset.sea_level) == 37
        check2 = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_HW' ).values == 8.01
        check3 = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_1' ).time.values == np.datetime64('2020-10-13 14:36')
        check4 = np.array_equal( tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_2' ).values, [2.83, 8.01] )
        check5 = np.array_equal( tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='window', winsize=24 ).values,  [3.47, 7.78, 2.8 , 8.01, 2.83, 8.45, 2.08, 8.71])

        assert check1
        assert check2
        assert check3
        assert check4
        assert check5
