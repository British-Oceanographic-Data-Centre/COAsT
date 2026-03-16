import pytest
import numpy as np
import xarray as xr

from coast._utils.time_stretch import get_start_year, get_end_year, get_date_range, extend_number_of_days, stretch_time

TIME_VAR_NAME = "time"
DATA_VAR_NAME = "test_variable"


@pytest.fixture  # Mock datasets.
def mock_dataset(request):
    """Fixture to return mock datasets."""
    if request.param == "mock_daily_dataset":
        times = xr.cftime_range("1850-01-01 12:00:00", "1851-01-01 00:00:00", freq="24H", calendar="360_day")
        data = np.random.rand(360)
        data_var = xr.DataArray(data=data, coords={TIME_VAR_NAME: times})
        return xr.Dataset(data_vars={DATA_VAR_NAME: data_var})
    if request.param == "mock_3hr_dataset":
        times = xr.cftime_range(
            "1850-01-01 03:00:00",
            "1851-01-01 00:00:00",
            freq="3H",
            calendar="360_day",
        )
        data = np.random.rand(360 * 8)
        data_var = xr.DataArray(data=data, coords={TIME_VAR_NAME: times})
        return xr.Dataset(data_vars={DATA_VAR_NAME: data_var})
    if request.param == "mock_non_leap_dataset":
        times = xr.cftime_range(
            "1904-01-01 06:00:00",
            "1905-01-01 00:00:00",
            freq="6H",
            calendar="noleap",
        )
        data = np.random.rand(365 * 4)
        data_var = xr.DataArray(data=data, coords={TIME_VAR_NAME: times})
        return xr.Dataset(data_vars={DATA_VAR_NAME: data_var})


@pytest.mark.parametrize("mock_dataset", [("mock_daily_dataset"), ("mock_3hr_dataset")], indirect=["mock_dataset"])
def test_get_start_year(mock_dataset):
    """Test method to return start year of dataset."""
    start_year = get_start_year(mock_dataset[TIME_VAR_NAME])
    assert start_year == 1850


@pytest.mark.parametrize("mock_dataset", [("mock_daily_dataset"), ("mock_3hr_dataset")], indirect=["mock_dataset"])
def test_get_end_year(mock_dataset):
    """Test method to return end year of dataset."""
    end_year = get_end_year(mock_dataset[TIME_VAR_NAME])
    assert end_year == 1851


@pytest.mark.parametrize(
    "mock_dataset, frequency", [("mock_daily_dataset", 24), ("mock_3hr_dataset", 3)], indirect=["mock_dataset"]
)
def test_get_date_range(mock_dataset, frequency):
    """Test method to generate standard calendar date ranges."""
    date_range = get_date_range(mock_dataset[TIME_VAR_NAME], frequency)
    assert len(date_range) == 365 * 24 / frequency


@pytest.mark.parametrize(
    "mock_dataset, frequency, target_days",
    [("mock_daily_dataset", 24, 365), ("mock_3hr_dataset", 3, 365), ("mock_non_leap_dataset", 6, 366)],
    indirect=["mock_dataset"],
)
def test_extend_number_of_days(mock_dataset, frequency, target_days):
    """Test method to generate an exteneded time array for a given time array."""
    points_in_data = int(len(mock_dataset[TIME_VAR_NAME]))
    measures_per_day = 24 / frequency
    extra_days = target_days - (points_in_data / measures_per_day)
    extended_time = extend_number_of_days(points_in_data, measures_per_day, extra_days)
    assert len(extended_time) == (target_days * measures_per_day)
    assert extended_time[0] == 1
    assert (
        extended_time[-1] == points_in_data
    )  # Time exension should never exceed original points in data per day (so we can interpolate with these new values.)


@pytest.mark.parametrize(
    "mock_dataset, frequency, target_days",
    [("mock_daily_dataset", 24, 365), ("mock_3hr_dataset", 3, 365), ("mock_non_leap_dataset", 6, 366)],
    indirect=["mock_dataset"],
)
def test_stretch_time(mock_dataset, frequency, target_days):
    """Test method to strecth time of a given xarray dataset."""
    stretched_ds = stretch_time(mock_dataset, hourly_interval=frequency)
    assert isinstance(stretched_ds, xr.Dataset)
    assert DATA_VAR_NAME in stretched_ds.variables
    assert TIME_VAR_NAME in stretched_ds.variables
    assert len(stretched_ds.variables) == 2  # Time and var1
    assert len(stretched_ds[TIME_VAR_NAME]) == target_days * (24 / frequency)
    assert len(stretched_ds[DATA_VAR_NAME]) == target_days * (24 / frequency)
    assert np.equal(
        mock_dataset[DATA_VAR_NAME].data[0:15], stretched_ds[DATA_VAR_NAME].data[0:15]
    ).all()  # First 15 elements unchanged.
    assert np.equal(
        mock_dataset[DATA_VAR_NAME].data[-15:-1], stretched_ds[DATA_VAR_NAME].data[-15:-1]
    ).all()  # Last 15 elements unchanged.
