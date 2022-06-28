import traceback
from typing import List
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


def get_start_year(time_data: np.ndarray) -> int:
    """Returns the start date of the dataset.

    Args:
        time_data (np.ndarray): Array of time data.

    Returns:
        int: Start date.
    """
    return time_data[0].dt.year.data


def get_end_year(time_data: np.ndarray) -> int:
    """Returns the end date of the dataset.

    Args:
        time_data (np.ndarray): Array of time data.

    Returns:
        int: End date.
    """
    last_day = time_data[-1]
    if last_day.dt.month.data == 1 and last_day.dt.day.data == 1:
        return last_day.dt.year.data - 1
    return last_day.dt.year.data


def get_date_range(time_data: np.ndarray, hourly_interval) -> List:
    """Return a date range for given time data and an hourly interval.

    Args:
        time_data (np.ndarray): Array of time data.
        hourly_interval (_type_): Hourly interval of new range.

    Returns:
        List: List of dates within a range.
    """
    return pd.date_range(
        f"{get_start_year(time_data)}-01-01 00:00:00",
        f"{get_end_year(time_data)}-12-31 23:59:59",
        freq=f"{hourly_interval}H",
    ).tolist()


def extend_number_of_days(points_in_data: int, measures_per_day: int, extra_days: int) -> np.ndarray:
    """Method to generate array for indexing of extended time.

    Args:
        points_in_data (int): Number of data points in data.
        measures_per_day (int): Number of data measurements per day.
        extra_days (int): Number of extra days to add to data.

    Returns:
        np.ndarray: Array of integers to represent extended time.
    """
    # Expand 330 days to 335/336 days (365 days - 30 ignored days.)

    first_15 = np.arange(1, (15 * measures_per_day) + 1)  # Ignore first 15 days.
    last_15 = np.arange(points_in_data - (15 * measures_per_day) + 1, points_in_data + 1)  # Ignore last 15 days.
    extended_days = np.arange(
        (15 * measures_per_day) + 1,
        points_in_data - (15 * measures_per_day) + 1,
        (points_in_data - (30 * measures_per_day))
        / (points_in_data - (30 * measures_per_day) + (extra_days * measures_per_day)),
    )
    extended_time = np.append(np.append(first_15, extended_days), last_15)

    return extended_time


def add_time(dataset: xr.Dataset, time_var_name: str = "time", hourly_interval: int = 24) -> xr.Dataset:
    """Method to stretch time and interpolate data for new time index.

    Args: 
        dataset (xr.Dataset): Original dataset.
        time_var_name (str, optional): Name of time variable within dataset. Defaults to "time".
        hourly_interval (int, optional): Interval of data measurements in hours. Defaults to 24.

    Returns:
        xr.Dataset : New dataset with stretched time values and interpolated data.
    """

    time_data = dataset[time_var_name]
    date_range = get_date_range(time_data, hourly_interval)
    measures_per_day = 24 / hourly_interval
    days = len(date_range) / measures_per_day

    points_in_data = int(len(time_data))

    time_original = np.arange(1, points_in_data + 1, 1)
    extra_days = days - (points_in_data / measures_per_day)

    extended_time = extend_number_of_days(
        points_in_data=points_in_data, measures_per_day=measures_per_day, extra_days=extra_days
    )

    stretched_variables = []
    for var_name, data_var in dataset.variables.items():
        if time_var_name not in data_var.dims:
            continue
        try:

            # Interpolate data.
            interpolate1d = interp1d(
                time_original, y=data_var[:], axis=0
            )  # Scipy interp retains all dims. (No np.squeeze)
            new_data = interpolate1d(extended_time)

            # Create new stretched variable.
            dim_dict = {dim: dataset[dim][:] for dim in data_var.dims if dim != time_var_name}
            dim_dict[time_var_name] = xr.cftime_range(
                start = str(dataset[time_var_name][0].dt.strftime("%Y-%m-%d %H:%M:%S").data),
                end = str(dataset[time_var_name][-1].dt.strftime("%Y-%m-%d %H:%M:%S").data),
                periods = extended_time.size,
                freq = None,
                calendar="all_leap"
            )
            # Create new data array for stretched variable.
            data_array = xr.DataArray(data=new_data, coords=dim_dict, name=var_name, dims=data_var.dims)
            stretched_variables.append(data_array)
        except Exception:
            print(f"{var_name} -- {traceback.format_exc()}")
    # Create new dataset from all stretched variables.
    new_dataset = xr.Dataset(data_vars={var.name: var for var in stretched_variables if var.name != time_var_name })
    return new_dataset
