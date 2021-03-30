from .COAsT import COAsT
import numpy as np
import xarray as xr
import xarray.ufuncs as uf
from dask.diagnostics import ProgressBar


class CLIMATOLOGY(COAsT):
    '''
    A Python class containing methods for lazily creating climatologies of
    NEMO data (or any xarray datasets) and writing to file. Also for resampling
    methods.
    '''
    
    def __init__(self):
        return
    
    @staticmethod
    def make_climatology(ds, output_frequency, monthly_weights = False, 
                              time_var_name = 'time', time_dim_name = 't_dim',
                              fn_out = None, missing_values = False):
        '''
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
        '''
        
        frequency_str = time_var_name + '.' + output_frequency
        print('Calculating climatological mean')
        
        if missing_values:
            ds_mean=xr.Dataset()
            for varname, da in ds.data_vars.items():       
                mask = xr.where(uf.isnan(da), 0, 1 )
                data = da.groupby(frequency_str).sum(dim=time_dim_name) 
                N = mask.groupby(frequency_str).sum(dim=time_dim_name)
                ds_mean[varname] = data / N
        else:
            if monthly_weights:
                month_length = ds[time_var_name].dt.days_in_month
                grouped = month_length.groupby(frequency_str)
            else:
                ds['clim_mean_ones_tmp'] = (time_dim_name, np.ones(ds[time_var_name].shape[0]))
                grouped = ds['clim_mean_ones_tmp'].groupby(frequency_str)

            weights = grouped / grouped.sum()
            ds_mean = (ds*weights).groupby(frequency_str).sum(dim=time_dim_name)

            if not monthly_weights:
                ds = ds.drop_vars('clim_mean_ones_tmp')

        if fn_out is not None:
            print('Saving to file. May take some time..')
            with ProgressBar():
                ds_mean.to_netcdf(fn_out)
        
        return ds_mean
        
        return