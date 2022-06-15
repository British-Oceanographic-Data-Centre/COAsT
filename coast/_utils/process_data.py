from ..data.gridded import Gridded
from .logging_util import info, warn, error
import dask.array as da
from dask import delayed
import xarray as xr
import numpy as np


class Process_data:  # TODO All abstract methods should be implemented
    """
    A Python class containing methods for processing time series.
    """

    def __init__(self):
        return  # TODO Super __init__ should be called at some point

    def seasonal_decomp(self, ts_chunk, **kwargs):
        """
        Note this is intended as an inner functions to be called by a wrapper function
        Process_data.seasonal_decomposition

        This function is itself a wrapper function for statsmodel.seasonal_decompose
        that accepts multiple timeseries as multiple columns (s_dim)
        as an xr.DataArray: ts_chunk = ts_chunk(t_dim,s_dim).
        Returns the trend, seasonal and residual componenets of the time series.
        Invalid points i.e. land points, should be specified as np.nans.

        If called directly, evaluation is eager, i.e. numpy arrays are returned
        for the trend, seasonal and residual components of the time series

        ts_chunk is the input array or chunk
        **kwargs are the kwargs for statsmodels.seasonal_decompose

        Calling with dask.delayed:
        dask.delayed(self.seasonal_decomp, nout=3)(ts_chunk, **kwargs)

        Need statsmodels package installed
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Get the indices of land points (points that are nan for all time steps)
        active_ind = np.where(np.nan_to_num(ts_chunk.data).any(axis=0))[0]
        # Remove them (statsmodel.seasonal_decompose will not accept them)
        ts_chunk_valid = ts_chunk[:, active_ind]
        # Call statsmodels.seasonal_decompose()
        decomp_chunk = seasonal_decompose(ts_chunk_valid, **kwargs)
        # Store time series components in seperate arrays and return
        trend_chunk, seasonal_chunk, residual_chunk = (
            np.full(ts_chunk.shape, np.nan),
            np.full(ts_chunk.shape, np.nan),
            np.full(ts_chunk.shape, np.nan),
        )
        trend_chunk[:, active_ind] = decomp_chunk.trend
        seasonal_chunk[:, active_ind] = decomp_chunk.seasonal
        residual_chunk[:, active_ind] = decomp_chunk.resid
        return trend_chunk, seasonal_chunk, residual_chunk

    def seasonal_decomposition(self, time_series: xr.DataArray, num_chunks=1, **kwargs):
        """
        Dask delayed wrapper for statsmodel.seasonal_decompose that accepts multiple timeseries
        distributed across multiple dimensitons as an xr.DataArray:
        time_series = time_series(t_dim,z_dim,x_dim,y_dim,...)

        Accepts land points (z,x,y) as time_series=np.nan for all time.

        num_chunks allows caller to specify the number of chunks in the delayed call, which enables
        parallelism. i.e. num_chunks = 4 splits the timeserieses into 4 groups that will be decomposed
        in parallel by the dask scheduler.

        **kwargs are the kwargs for statsmodels.seasonal_decompose function

        Returns coast.Gridded object containing an xarray.Dataset with the trend, seasonal, residual
        timeseries components as xarray.DataArrays.

        Need statsmodels package installed
        """
        # first dim needs to be time
        time_series = time_series.transpose("t_dim", ...)
        # stack all the other dims
        ts_stacked = time_series.stack(space=time_series.dims[1:])
        # calculate chunk size for stacked dimension and rechunk
        chunk_size = ts_stacked["space"].size // num_chunks + 1
        ts_stacked = ts_stacked.chunk({"t_dim": ts_stacked["t_dim"].size, "space": chunk_size})
        # convert to list of dask delayed objects (e.g. 4 chunks => 4 delayed objects in list)
        ts_delayed = ts_stacked.data.to_delayed().ravel()

        # delayed call to statsmodel.seasonal_decomposition (with some other processing) that returns
        # 3 lists:
        # 1) contains the dask delayed chunks for trend (e.g. 4 delayed objects in list if 4 chunks),
        # 2) contains the dask delayed chunks for seasonal,
        # 3) contains the dask delayed chunks for residual,
        trend_de, seasonal_de, resid_de = map(
            list, zip(*[delayed(self.seasonal_decomp, nout=3)(ts_chunk, **kwargs) for ts_chunk in ts_delayed])
        )

        # convert to lists of dask arrays to allow array operations
        trend_da, seasonal_da, resid_da = [], [], []
        for chunk_idx, (tr_chunk, se_chunk, re_chunk) in enumerate(zip(trend_de, seasonal_de, resid_de)):
            # When converting from delayed to array, you must know the shape of the
            # array. Here we know this from the chunk sizes of the (stacked) DataArray
            chunk_shape = (ts_stacked.chunks[0][0], ts_stacked.chunks[1][chunk_idx])
            trend_da.append(da.from_delayed(tr_chunk, shape=chunk_shape, dtype=float))
            seasonal_da.append(da.from_delayed(se_chunk, shape=chunk_shape, dtype=float))
            resid_da.append(da.from_delayed(re_chunk, shape=chunk_shape, dtype=float))

        # concatenate the array chunks together and reshape back to original shape
        trend = da.reshape(da.concatenate(trend_da, axis=1), time_series.shape)
        seasonal = da.reshape(da.concatenate(seasonal_da, axis=1), time_series.shape)
        residual = da.reshape(da.concatenate(resid_da, axis=1), time_series.shape)
        # create  gridded object for return. New DataArrays created from the original
        # timeseries DataArray so it will have the smae coordinates and shape. Rechunk it
        # to have same structure as the newly created dask arrays and then assign to gridded
        gd = Gridded()
        gd.dataset["trend"] = xr.full_like(time_series.chunk(trend.chunks), np.nan)
        gd.dataset["trend"][:] = trend
        gd.dataset["seasonal"] = xr.full_like(time_series.chunk(seasonal.chunks), np.nan)
        gd.dataset["seasonal"][:] = seasonal
        gd.dataset["residual"] = xr.full_like(time_series.chunk(residual.chunks), np.nan)
        gd.dataset["residual"][:] = residual
        return gd
