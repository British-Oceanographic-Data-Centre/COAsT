from .COAsT import COAsT
import numpy as np
from .logging_util import get_slug, debug, info, warn, warning, error
import xarray as xr


class CLIMATOLOGY(COAsT):
    def __init__(self):
        return
    
    def make_climatology(self, dataset, frequency, time_var = "time", 
                         time_dim = "t_dim"):
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
            'dayofyear'
            'hour'
        time_var :: the string name of the time variable in dataset
        time_dim :: the string name of the time dimension variable in dataset
        '''
        
        # For the sake of it, if an xarray dataset is not provided, check to
        # see if there is a dataset inside it (like a NEMO object)
        if type(dataset) != xr.core.dataset.Dataset:
            dataset = dataset.dataset
        
        # Group the dataset by frequency
        clim = dataset.groupby(time_var+"."+frequency)
        
        # Calculate the mean and variance
        mean = clim.mean(time_var)
        var = clim.var(time_var)
        
        return mean, var
    
    def to_netcdf_in_slices():
        return
    
