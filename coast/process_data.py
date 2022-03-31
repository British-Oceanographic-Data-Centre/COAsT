from .gridded import Gridded
from .logging_util import info, warn, error
import dask.array as da
from dask import delayed
import xarray as xr
import numpy as np



class Process_data:  # TODO All abstract methods should be implemented
    """
    A Python class containing methods for lazily creating climatologies of
    NEMO data (or any xarray datasets) and writing to file. Also for resampling
    methods.
    """

    def __init__(self):
        return  # TODO Super __init__ should be called at some point

#    def seasonal_decomposition_np(self, data_chunk, **kwargs ):
#        from statsmodels.tsa.seasonal import seasonal_decompose
#        #model='additive', period=12, extrapolate_trend="freq"
#        result_stacked = seasonal_decompose(data_chunk, **kwargs)  
#        return result_stacked 
    
    def _remove_invalid(self, data_stacked):
        active_ind = np.where(np.nan_to_num(data_stacked.data).any(axis=0))[0] 
        return data_stacked[:,active_ind]

    def _rebuild_component(self, component_chunks: list, ts_shape, ts_stacked_shape, active_ind):
        component = np.full(ts_stacked_shape, np.nan)    
        component[:, active_ind] = np.concatenate(component_chunks, axis = 1)
        return np.reshape(component, ts_shape).squeeze()
    
    def seasonal_decomp(self, ts_chunk, **kwargs):    
        from statsmodels.tsa.seasonal import seasonal_decompose
        active_ind = np.where(np.nan_to_num(ts_chunk.data).any(axis=0))[0] 
        ts_chunk_valid = ts_chunk[:,active_ind]        
        #model='additive', period=12, extrapolate_trend="freq"
        decomp_chunk = seasonal_decompose(ts_chunk_valid, **kwargs)
        trend_chunk, seasonal_chunk, residual_chunk = np.full(ts_chunk.shape, np.nan), np.full(ts_chunk.shape, np.nan), np.full(ts_chunk.shape, np.nan)    
        trend_chunk[:, active_ind] = decomp_chunk.trend 
        seasonal_chunk[:, active_ind] = decomp_chunk.seasonal
        residual_chunk[:, active_ind] = decomp_chunk.resid
        return trend_chunk, seasonal_chunk, residual_chunk
    
    def seasonal_decomposition(self, time_series: xr.DataArray, num_chunks=1, **kwargs):
        # first dim needs to be time
        time_series = time_series.transpose("t_dim",...)
        # stack all the other dims
        ts_stacked = time_series.stack(space=time_series.dims[1:])
        # calculate chunk size for stacked dimension and rechunk
        chunk_size = ts_stacked["space"].size // num_chunks + 1
        ts_stacked = ts_stacked.chunk({"t_dim": ts_stacked["t_dim"].size, "space": chunk_size})
        # convert to list of dask delayed objects (e.g. 4 chunks => 4 delayed objects in list)
        ts_delayed = ts_stacked.data.to_delayed().ravel()
        
        # delayed call to statsmodel.seasonal_decomposition (with some other processing) that returns
        # a 3 lists: 
        # 1) contains the dask delayed chunks for trend, 
        # 2) contains the dask delayed chunks for seasonal, 
        # 3) contains the dask delayed chunks for residual,
        trend_de, seasonal_de, resid_de = map(
            list,zip(*[delayed(self.seasonal_decomp, nout=3)(ts_chunk, **kwargs) for ts_chunk in ts_delayed])
        )
        
        # convert to lists of dask arrays to allow array operations
        trend_da, seasonal_da, resid_da = [], [], []
        for chunk_idx, (tr_chunk, se_chunk, re_chunk) in enumerate(zip(trend_de, seasonal_de, resid_de)):
            chunk_shape = (ts_stacked.chunks[0][0], ts_stacked.chunks[1][chunk_idx])                                      
            trend_da.append(da.from_delayed(tr_chunk, shape=chunk_shape, dtype=float))
            seasonal_da.append(da.from_delayed(se_chunk, shape=chunk_shape, dtype=float))
            resid_da.append(da.from_delayed(re_chunk, shape=chunk_shape, dtype=float))
        
        gd = Gridded()
        trend = da.reshape( da.concatenate( trend_da, axis=1 ), time_series.shape )
        seasonal = da.reshape( da.concatenate( seasonal_da, axis=1 ), time_series.shape )
        residual = da.reshape( da.concatenate( resid_da, axis=1 ), time_series.shape )
        gd.dataset["trend"] = xr.full_like(time_series.chunk(trend.chunks), np.nan)
        gd.dataset["trend"][:,:,:] = trend
        gd.dataset["seasonal"] = xr.full_like(time_series.chunk(seasonal.chunks), np.nan)
        gd.dataset["seasonal"][:,:,:] = seasonal
        gd.dataset["residual"] = xr.full_like(time_series.chunk(residual.chunks), np.nan)
        gd.dataset["residual"][:,:,:] = residual
        return gd

#    def seasonal_decomposition(self, time_series: xr.DataArray, num_chunks=1, **kwargs):
#        time_series = time_series.transpose("t_dim",...)
#        ts_stacked = time_series.stack(space=time_series.dims[1:])
#        
#        #active_ind = xr.where( 
#        #    ts_stacked.fillna(0).any(dim="t_dim"), 
#        #    da.arange(0, ts_stacked.space.size), 
#        #    np.nan 
#        #).dropna("space").astype(int)
#        #ts_stacked_valid = ts_stacked.isel( space=active_ind )
#        
#        ts_stacked_valid = delayed(self._remove_invalid)(ts_stacked)        
#        
#        chunk_size = ts_stacked_valid["space"].size // num_chunks + 1
#        ts_stacked_valid = ts_stacked_valid.chunk({"t_dim": ts_stacked_valid["t_dim"].size, "space": chunk_size})
#        
#        decomp_results = [
#            delayed(self.seasonal_decomposition_np)(ts_chunk, **kwargs) 
#            #for ts_chunk in ts_stacked_valid.data.to_delayed().ravel()
#            for ts_chunk in ts_stacked_valid.data.ravel()
#        ]    
#
#        trend_chunks, seasonal_chunks, residual_chunks = [], [], []
#        for decomp_results_chunk in decomp_results:
#            trend_chunks.append(decomp_results_chunk.trend)
#            seasonal_chunks.append(decomp_results_chunk.seasonal)
#            residual_chunks.append(decomp_results_chunk.resid)
#
#        ts_shape = time_series.shape
#        trend_delayed = delayed(self._rebuild_component)(trend_chunks, ts_shape, ts_stacked.shape, active_ind)
#        seasonal_delayed = delayed(self._rebuild_component)(seasonal_chunks, ts_shape, ts_stacked.shape, active_ind)
#        residual_delayed = delayed(self._rebuild_component)(residual_chunks, ts_shape, ts_stacked.shape, active_ind)
#
#        gd_out = Gridded()
#        #gd_out.dataset["trend"] = xr.full_like(time_series.chunk(ts_shape), np.nan)
#        #gd_out.dataset["trend"][:,:,:] = da.from_delayed(trend_delayed, ts_shape, dtype=float)
#        #gd_out.dataset["seasonal"] = xr.full_like(time_series.chunk(ts_shape), np.nan)
#        #gd_out.dataset["seasonal"][:,:,:] = da.from_delayed(seasonal_delayed, ts_shape, dtype=float)
#        #gd_out.dataset["residual"] = xr.full_like(time_series.chunk(ts_shape), np.nan)
#        #gd_out.dataset["residual"][:,:,:] = da.from_delayed(residual_delayed, ts_shape, dtype=float)
#        return gd_out  
    
"""
def seasonal_decomposition(time_series: xr.DataArray):

#    time series must be hozizontal 2d
    

    
    from statsmodels.tsa.seasonal import seasonal_decompose
                           
    data_stacked = time_series.fillna(0).stack(space=("x_dim","y_dim"))
    active_ind = np.where( np.nan_to_num(data_stacked).any(axis=0))[0] 
    data_stacked_valid = data_stacked[:,active_ind]
    result_stacked = seasonal_decompose(data_stacked_valid, model='additive', period=12, extrapolate_trend="freq")

        trend = np.empty(np.shape(data_stacked))
    seasonal = np.empty(np.shape(data_stacked))
    residual = np.empty(np.shape(data_stacked))

    trend[:,active_ind] = result_stacked.trend
    seasonal[:,active_ind] = result_stacked.seasonal
    residual[:,active_ind] = result_stacked.resid

    gd_out = coast.Gridded()
    gd_out.dataset["trend"] = xr.full_like(da_in, np.nan)
    gd_out.dataset["trend"][:,:,:] = trend.reshape(np.shape(da_in))
    gd_out.dataset["seasonal"] = xr.full_like(da_in, np.nan)
    gd_out.dataset["seasonal"][:,:,:] = seasonal.reshape(np.shape(da_in))
    gd_out.dataset["residual"] = xr.full_like(da_in, np.nan)
    gd_out.dataset["residual"][:,:,:] = residual.reshape(np.shape(da_in))
    """