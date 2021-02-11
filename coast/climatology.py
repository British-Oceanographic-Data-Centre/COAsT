from .COAsT import COAsT
import numpy as np
from .logging_util import get_slug, debug, info, warn, warning, error
import xarray as xr
import pandas as pd

class CLIMATOLOGY(COAsT):
    def __init__(self):
        return
    
    def make_climatology(self, da, frequency, calculate_var = False):
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
        
        dataset :: xarray dataset object from a COAsT object.
        frequency :: any xarray groupby string. i.e:
            'month'
            'season'
        time_var :: the string name of the time variable in dataset
        time_dim :: the string name of the time dimension variable in dataset
        '''
        
        n_time_dict = {"month":12, "season":4}
        month_season_dict = {1:4, 2:1, 3:1, 4:1, 5:2, 6:2,
                             7:2, 8:3, 9:3, 10:3, 11:4, 12:4}
        sum_array = np.zeros([n_time_dict[frequency], len(da.z_dim), 
                         len(da.y_dim), len(da.x_dim)])
        N_weights = np.zeros(n_time_dict[frequency])
        
        
        for tt in range(0,len(da.t_dim)):
            print(tt)
            snapshot = da.isel(t_dim=tt).load()
            dt_time = pd.to_datetime(snapshot.time.values)
            if frequency=='month':
                time_index = dt_time.month - 1
            elif frequency=='season':
                time_index=month_season_dict(dt_time.month) - 1
                
            if tt == 0:
                sum_array[time_index] = snapshot
            else:
                sum_array[time_index] = sum_array[time_index] + snapshot
            N_weights[time_index] = N_weights[time_index] + 1
            
        if calculate_var:
            pass
                
        for tt in range(0,len(da.t_dim)):
            print(tt)
            sum_array[tt] = sum_array[tt]/N_weights[tt]
            
        sum_array = xr.Dataset(data_vars = dict(
                         mean = ([frequency, 'z_dim','y_dim','x_dim'], sum_array),
                     ),
                     coords = dict(
                         longitude=da.longitude,
                         latitude=da.latitude,
                     ))
            
        return sum_array
            
            
        
    
    def to_netcdf_in_slices():
        return
    
