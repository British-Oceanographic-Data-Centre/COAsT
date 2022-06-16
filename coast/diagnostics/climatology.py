"""Climatology class"""
import calendar
from datetime import date
import traceback
from typing import List, Tuple

from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import xarray as xr

from ..data.coast import Coast
from .._utils.logging_util import info, warn, error


class Climatology(Coast):  # TODO All abstract methods should be implemented
    """
    A Python class containing methods for lazily creating climatologies of
    NEMO data (or any xarray datasets) and writing to file. Also for resampling
    methods.
    """

    def __init__(self):
        return  # TODO Super __init__ should be called at some point

    @staticmethod
    def make_climatology(
        ds, output_frequency, monthly_weights=False, time_var_name="time", time_dim_name="t_dim", fn_out=None
    ):
        """
        Calculates a climatology for all variables in a supplied dataset.
        The resulting xarray dataset will NOT be loaded to RAM. Instead,
        it is a set of dask operations. To load to RAM use, e.g. .compute().
        However, if the original data was large, this may take a long time and
        a lot of memory. Make sure you have the available RAM or chunking
        and parallel processes are specified correctly.

        Otherwise, it is recommended that you access the climatology data
        in an indexed way. I.E. compute only at specific parts of the data
        are once.

        The resulting cliamtology dataset can be written to disk using
        .to_netcdf(). Again, this may take a while for larger datasets.

        ds :: xarray dataset object from a Coast object.
        output_frequency :: any xarray groupby string. i.e:
            'month'
            'season'
        time_var_name :: the string name of the time variable in dataset
        time_dim_name :: the string name of the time dimension variable in dataset
        fn_out :: string defining full output netcdf file path and name.
        """

        frequency_str = time_var_name + "." + output_frequency
        info("Calculating climatological mean")

        if monthly_weights:
            month_length = ds[time_var_name].dt.days_in_month
            grouped = month_length.groupby(frequency_str)
            weights = grouped / grouped.sum()
            ds_mean = (ds * weights).groupby(frequency_str).sum(dim=time_dim_name)
        else:
            ds_mean = xr.Dataset()
            for var_name, da in ds.data_vars.items():
                try:
                    da_mean = da.groupby(frequency_str).mean(dim=time_dim_name, skipna=True)
                    ds_mean[var_name] = da_mean
                except ArithmeticError:
                    error(f"Skipped mean calculation for {var_name} due to error: {traceback.format_exc()}")

        if fn_out is not None:
            info("Saving to file. May take some time..")
            with ProgressBar():
                ds_mean.to_netcdf(fn_out)

        return ds_mean

    @staticmethod
    def _get_date_ranges(years: List[int], month_periods: List[Tuple[int, int]]) -> List[Tuple[date, date]]:
        """Calculates a list of datetime date ranges for a given list of years and a specified start/end month.

        Args:
            years (list): A list of years to calculate date ranges for.
            month_periods (list): A list containing tuples of start and end month integers.
            (i.e. [(3,5),(12, 2)] is Mar -> May, Dec -> Feb). Must be in chronological order.

            Returns:
                date_ranges (list): A list of tuples, each containing a start and end datetime.date object.
        """
        date_ranges = []
        for y in sorted(set(years)):
            y = int(y)
            for period in month_periods:
                start = period[0]
                end = period[1]
                begin_date = date(y, start, 1)
                if start > end:
                    end_day = calendar.monthrange(y + 1, end)[1]
                    end_date = date(y + 1, end, end_day)
                else:
                    end_day = calendar.monthrange(y, end)[1]
                    end_date = date(y, end, end_day)
                date_ranges.append((begin_date, end_date))
        return date_ranges

    @classmethod
    def multiyear_averages(
        cls, ds: xr.Dataset, month_periods: List[Tuple[int, int]], time_var: str = "time", time_dim: str = "t_dim"
    ) -> xr.Dataset:
        """Calculate multiyear means for all Data variables in a dataset between a given start and end month.

        Args:
            ds (xr.Dataset): xarray dataset containing data.
            month_periods (list): A list containing tuples of start and end month integers.
            (i.e. [(3,5),(12, 2)] is Mar -> May, Dec -> Feb). Must be in chronological order.
            The seasons module can be used for convenience (e.g. seasons.WINTER, seasons.ALL etc. )
            time_var (str): String representing the time variable name within the dataset.
            time_dim (str): String representing the time dimension name within the dataset.
        returns:
            ds_mean (xr.Dataset): A new dataset containing mean averages for each data variable across all years and
            date periods. Indexed by the multi-index 'year_period' (i.e. (2000, 'Dec-Feb')).
        """

        time_dim_da = ds[f"{time_dim}"]
        time_var_da = ds[f"{time_var}"]

        new_ds = ds
        # If time dimension isn't np.datetime64 but time variable is, then swap time variable to be the dimension.
        # There should be a 1 to 1 mapping between time dimension values and time variable values.
        # A datetime type is required for slicing dates over a dimension using xarray's sel() method.
        if not np.issubdtype(time_dim_da.dtype, np.datetime64) and np.issubdtype(time_var_da.dtype, np.datetime64):
            warn("Time dimension is not np.datatime64 but time variable is. Swapping time dimension for data variable.")
            # Swap time_var with time_dim.
            new_ds = ds.swap_dims({f"{time_dim}": f"{time_var}"})
            time_dim = time_var
        elif not np.issubdtype(time_dim_da.dtype, np.datetime64) and not np.issubdtype(
            time_var_da.dtype, np.datetime64
        ):
            # Slicing will most likely fail for non np.datetime64 datatypes.
            warn("Neither time dimension or time variable data are np.datetime64. Time slicing may fail.")

        # Get years of data.
        data_years = list(new_ds[f"{time_dim}.year"].data)
        # Append first year - 1, to account for possible WINTER month data of year prior to data beginning.
        data_years.insert(0, data_years[0] - 1)

        # Generate date ranges from years and given month periods.
        date_ranges = Climatology._get_date_ranges(data_years, month_periods)

        # Extract data from dataset between these date ranges and index each range with a common multi-index.
        datasets = []
        year_index = []
        month_index = []
        for date_range in date_ranges:
            sel_args = {f"{time_dim}": slice(date_range[0], date_range[1])}
            filtered = new_ds.sel(**sel_args)
            datasets.append(filtered)
            year_index = year_index + ([date_range[0].year] * filtered.sizes[time_dim])
            month_label = f"{calendar.month_abbr[date_range[0].month]}-{calendar.month_abbr[date_range[1].month]}"
            month_index = month_index + ([month_label] * filtered.sizes[time_dim])

        # New dataset built from extracted data between date ranges.
        filtered = xr.concat(datasets, dim=time_dim)
        # Data from same date range use common year-period multi-index so they can be grouped together.
        period_idx = pd.MultiIndex.from_arrays([year_index, month_index], names=("year", "period"))
        filtered.coords["year_period"] = (f"{time_dim}", period_idx)

        # For each data variable, group on year-period multi-index and find the mean.
        # New dataset containing means across date ranges is returned.
        ds_mean = xr.Dataset()
        for var_name, da in filtered.data_vars.items():
            try:
                # Apply .mean() to grouped data.
                # skipna flag used to ignore NaN values.
                da_mean = da.groupby("year_period").mean(dim=time_dim, skipna=True)
                ds_mean[f"{var_name}"] = da_mean
            except ArithmeticError:
                warn(f"Skipped mean calculation for {var_name} due to error: {traceback.format_exc()}")
        return ds_mean
