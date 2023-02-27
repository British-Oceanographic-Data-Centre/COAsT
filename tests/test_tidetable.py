# Test with PyTest

import coast
import numpy as np
import xarray as xr
import datetime
import logging
import coast._utils.general_utils as general_utils


# -----------------------------------------------------------------------------#
# %% day of the week function                                           #
#                                                                             #
def test_dayoweek():
    check1 = general_utils.day_of_week(np.datetime64("2020-10-16")) == "Fri"
    assert check1
