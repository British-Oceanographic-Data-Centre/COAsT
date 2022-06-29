import pytest
import numpy as np
import xarray as xr

from coast._utils.time_stretch import get_start_year, get_end_year, get_date_range


@pytest.fixture  # Mock datasets.
def mock_dataset(request):
    """Fixture to return mock datasets."""
    if request.param == "mock_daily_dataset":
        times = xr.cftime_range("1850-01-01 12:00:00", "1851-01-01 00:00:00", freq="24H", calendar="360_day")
        data = np.random.rand(360)
        data_var = xr.DataArray(data=data, coords={"time": times})
        return xr.Dataset(data_vars={"var1": data_var})
    if request.param == "mock_3hr_dataset":
        times = xr.cftime_range("1850-01-01 03:00:00", "1851-01-01 00:00:00", freq="3H", calendar="360_day",)
        data = np.random.rand(360 * 8)
        data_var = xr.DataArray(data=data, coords={"time": times})
        return xr.Dataset(data_vars={"var1": data_var})


@pytest.mark.parametrize("mock_dataset", [("mock_daily_dataset"), ("mock_3hr_dataset")], indirect=["mock_dataset"])
def test_get_start_year(mock_dataset):
    """Test method to return start year of dataset."""
    start_year = get_start_year(mock_dataset["time"])
    assert start_year == 1850


@pytest.mark.parametrize("mock_dataset", [("mock_daily_dataset"), ("mock_3hr_dataset")], indirect=["mock_dataset"])
def test_get_end_year(mock_dataset):
    """Test method to return end year of dataset."""
    end_year = get_end_year(mock_dataset["time"])
    assert end_year == 1851


@pytest.mark.parametrize(
    "mock_dataset, frequency", [("mock_daily_dataset", 24), ("mock_3hr_dataset", 3)], indirect=["mock_dataset"]
)
def test_get_date_range(mock_dataset, frequency):
    """Test method to generate standard calendar date ranges."""
    date_range = get_date_range(mock_dataset["time"], frequency)
    assert len(date_range) == 365 * 24 / frequency
