from datetime import date
import pytest

import numpy as np
import pandas as pd
import xarray as xr

from coast._utils import seasons
from coast.diagnostics.climatology import Climatology


YEARS = [2000, 2001]
PERIOD = seasons.ALL
# Date ranges for each season of 2000 and 2001.
DATE_RANGES = [
    (date(2000, 3, 1), date(2000, 5, 31)),
    (date(2000, 6, 1), date(2000, 9, 30)),
    (date(2000, 10, 1), date(2000, 11, 30)),
    (date(2000, 12, 1), date(2001, 2, 28)),
    (date(2001, 3, 1), date(2001, 5, 31)),
    (date(2001, 6, 1), date(2001, 9, 30)),
    (date(2001, 10, 1), date(2001, 11, 30)),
    (date(2001, 12, 1), date(2002, 2, 28)),
]

EXPECTED_MEANS = np.array([45.5, 152.5, 244.0, 319.5, 410.5, 517.5, 609.0, 684.5])


@pytest.fixture
def test_dataset():
    time = pd.date_range(start=DATE_RANGES[0][0], end=DATE_RANGES[-1][1], freq="D")
    ds = xr.Dataset({"data": ("time", np.arange(len(time))), "data_ones": ("time", np.ones(len(time))), "time": time})
    yield ds


def test_get_date_ranges():
    result = Climatology._get_date_ranges(YEARS, PERIOD)
    assert result == DATE_RANGES


# Simple test for calculating means on a known small dataset. Generated within test_dataset().
def test_multiyear_averages(test_dataset):
    ds_mean = Climatology.multiyear_averages(test_dataset, PERIOD, time_var="time", time_dim="time")
    # Assert ds_mean meaned data in equal to our precalculated EXPECTED_MEANS values.
    assert np.array_equal(ds_mean["data"], EXPECTED_MEANS)
    # Assert data_ones var is in output dataset, and it's meaned values are all 1.
    # This is mainly to check that the multiyear_average method works on all variables in the initial dataset.
    assert np.array_equal(ds_mean["data_ones"], np.ones(len(DATE_RANGES)))
    # Assert there are 8 year_period index values in ds_mean. (One for each DATE RANGE.)
    assert len(ds_mean["year_period"]) == len(DATE_RANGES)
    # Assert dataset years are all values defined within the YEARS list.
    dataset_years = set(ds_mean["year"].data)
    assert set(dataset_years).issubset(YEARS)
