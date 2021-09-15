from .COAsT import COAsT
import calendar
from datetime import date
import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as uf
from dask.diagnostics import ProgressBar

from .logging_util import get_slug, debug, info, warn, error


class CLIMATOLOGY(COAsT):
    """
    A Python class containing methods for lazily creating climatologies of
    NEMO data (or any xarray datasets) and writing to file. Also for resampling
    methods.
    """

    def __init__(self):
        return

    @staticmethod
    def make_climatology(
        ds,
        output_frequency,
        monthly_weights=False,
        time_var_name="time",
        time_dim_name="t_dim",
        fn_out=None,
        missing_values=False,
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

        ds :: xarray dataset object from a COAsT object.
        output_frequency :: any xarray groupby string. i.e:
            'month'
            'season'
        time_var_name :: the string name of the time variable in dataset
        time_dim_name :: the string name of the time dimension variable in dataset
        fn_out :: string defining full output netcdf file path and name.
        missing_values :: boolean where True indicates the data has missing values
            that should be ignored. Missing values must be represented by NaNs.
        """

        frequency_str = time_var_name + "." + output_frequency
        info("Calculating climatological mean")

        if missing_values:
            ds_mean = xr.Dataset()
            for var_name, da in ds.data_vars.items():
                try:
                    mask = xr.where(uf.isnan(da), 0, 1)
                    data = da.groupby(frequency_str).sum(dim=time_dim_name)
                    total_points = mask.groupby(frequency_str).sum(dim=time_dim_name)
                    ds_mean[var_name] = data / total_points
                except Exception as e:
                    error(f"Problem with {var_name}: {e}")
        else:
            if monthly_weights:
                month_length = ds[time_var_name].dt.days_in_month
                grouped = month_length.groupby(frequency_str)
            else:
                ds["clim_mean_ones_tmp"] = (time_dim_name, np.ones(ds[time_var_name].shape[0]))
                grouped = ds["clim_mean_ones_tmp"].groupby(frequency_str)

            weights = grouped / grouped.sum()
            ds_mean = (ds * weights).groupby(frequency_str).sum(dim=time_dim_name)

            if not monthly_weights:
                ds = ds.drop_vars("clim_mean_ones_tmp")

        if fn_out is not None:
            info("Saving to file. May take some time..")
            with ProgressBar():
                ds_mean.to_netcdf(fn_out)

        return ds_mean

    @staticmethod
    def _get_date_ranges(years, month_periods):
        """Calculates a list of datetime date ranges for a given list of years and a specified start/end month.

        Args:
            years (list): A list of years to calculate date ranges for.
            month_periods (list): A list containing tuples of start and end month integers.
            (i.e. [(3,5),(12, 2)] is Mar -> May, Dec -> Feb). Must be in chronological order.

            Returns:
                date_ranges (list): A list of tuples, each containing a start and end datetime.date object.
        """
        date_ranges = []
        for y in set(years):
            y = int(y)
            for period in month_periods:
                start = period[0]
                end = period[1]
                if start > end:
                    begin_date = date(y, start, 1)
                    end_day = calendar.monthrange(y + 1, end)[1]
                    end_date = date(y + 1, end, end_day)
                else:
                    begin_date = date(y, start, 1)
                    end_day = calendar.monthrange(y, end)[1]
                    end_date = date(y, end, end_day)
                date_ranges.append((begin_date, end_date))
        return date_ranges

    @staticmethod
    def multiyear_averages(ds: xr.Dataset, month_periods: list, time_var: str = "time", time_dim: str = "t_dim"):
        """Calculate multiyear means for all Data variables in a dataset between a given start and end month.

        Args:
            ds (xr.Dataset): xarray dataset containing data.
            month_periods (list): A list containing tuples of start and end month integers.
            (i.e. [(3,5),(12, 2)] is Mar -> May, Dec -> Feb). Must be in chronological order.
            The Season class can be used for convenience (e.g. Season.WINTER, Season.All etc. )
            time_var (str): String representing the time variable name within the dataset.
            time_dim (str): String representing the time dimension name within the dataset.
        returns:
            ds_mean (xr.Dataset): A new dataset containing mean averages for each data variable across all years.
            Indexed by the key 'year'.
        """

        time_dim_da = ds[f"{time_dim}"]
        time_var_da = ds[f"{time_var}"]

        # If time dimension isn't np.datetime64 but time variable is, then swap time variable to be the dimension.
        # A datetime type is required for slicing data using xarray's sel() method.
        if not np.issubdtype(time_dim_da.dtype, np.datetime64) and np.issubdtype(time_var_da.dtype, np.datetime64):
            warn("Time dimension is not np.datatime64 but time variable is. Swapping time dimension for data variable.")
            new_ds = ds.swap_dims({f"{time_dim}": f"{time_var}"})
            time_dim = time_var
        else:
            # Slicing will most likely fail for non np.datetime64 datatypes.
            warn("Neither time dimension or time variable data are np.datetime64. Time slicing may fail.")
            new_ds = ds

        # Get years of data.
        data_years = list(new_ds[f"{time_dim}.year"].to_numpy())
        # Append first year - 1, to account for possible WINTER month data of year prior to data beginning.
        data_years.insert(0, data_years[0] - 1)

        # Generate date ranges from years and given month periods.
        date_ranges = CLIMATOLOGY._get_date_ranges(data_years, month_periods)

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
            except ArithmeticError as e:
                warn(f"Skipped mean calculation for {var_name} due to error: {e}")
        return ds_mean


class Season:
    """Class with attributes defining month ranges for the four seasons.

    Note: Summer is defined as JJAS, as opposed to the meteorological seasons of JJA.
    """

    SPRING = [(3, 5)]
    SUMMER = [(6, 9)]
    AUTUMN = [(10, 11)]
    WINTER = [(12, 2)]
    ALL = [(3, 5), (6, 9), (10, 11), (12, 2)]
