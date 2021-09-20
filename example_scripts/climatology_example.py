"""
This file has three main example scripts:

1)  make_climatology_nemo():
    This script uses the COAsT package to calculate a climatological mean of an
    input NEMO dataset at a desired output frequency. Output will be written straight
    to file.

2)  make_climatology_ostia():
    This script uses COAsT and xarray to calculate a climatological mean of an
    input OSTIA dataset at a desired output frequency. Output will be written straight
    to file.

3) make_multiyear_climatology():
    This script uses the COAsT package to calculate a climatological mean of an
    input NEMO dataset at a desired output frequency, over multiple years.

COAsT and xarray should preserve any lazy loading and chunking. If defined
properly in the read function, memory issues can be avoided and parallel
processes will automatically be used.
*NOTE: In any xarray.open_dataset or xarray.open_mfdataset calls, make sure
you switch on Dask by defining chunks. At the least, pass the argument
chunks = {} OR chunks = 'auto'.
"""

import coast
from coast import seasons
import xarray as xr


def make_climatology_nemo():
    """Example code explaining the usage of coast.CLIMATOLOGY.make_climatology() with NEMO data.

    Calculates mean over a given period of time. This doesn't take different years into account, unless using the
    'years' frequency.
    """

    # Paths to a single or multiple NEMO files.
    fn_nemo_data = '/path/to/nemo/datafile.nc'
    # Set path for domain file if required.
    fn_nemo_domain = None
    # Define output filepath (optional)
    fn_out = "/path/to/output/nemo_seasonal.nc"

    # Define frequency -- Any xarray time string: season, month, etc
    climatology_frequency = 'season'

    # Read in multiyear data (This example uses NEMO data from a single file.)
    nemo_data = coast.NEMO(fn_data=fn_nemo_data, fn_domain=fn_nemo_domain, chunks={}).dataset
    # Select specific data variables.
    data = nemo_data[['temperature', 'ssh', 'salinity']]

    # Calculate the climatology and write to file.
    clim = coast.CLIMATOLOGY()
    clim_mean = clim.make_climatology(data, climatology_frequency,
                                      fn_out=fn_out)

    return clim_mean


def make_climatology_ostia():
    """Example code explaining the usage of coast.CLIMATOLOGY.make_climatology() with OSTIA data.

    Calculates mean over a given period of time. This doesn't take different years into account, unless using the
    'years' frequency.
    """

    # Define output file
    fn_out = "/path/to/output/ostia_seasonal_mean.nc"
    
    # Define frequency -- Any xarray time string: season, month, etc
    climatology_frequency = 'season'
    
    # Read in and format dataset (This example uses OSTIA data.)
    fn_ostia = "/path/to/ostia/*.nc"
    kelvin_to_celcius = -273.15
    data = xr.open_mfdataset(fn_ostia, chunks={}, concat_dim='time', parallel=True)
    data = data.rename({'analysed_sst': 'temperature'})
    data = data.rename({'time': 't_dim'})
    data['temperature'] = data.temperature + kelvin_to_celcius
    data.attrs = {}
    
    # Calculate the climatology and write to file.
    clim = coast.CLIMATOLOGY()
    clim_mean = clim.make_climatology(data, climatology_frequency,
                                      fn_out=fn_out)
    return clim_mean


def make_multiyear_climatology():
    """Example code explaining the usage of coast.CLIMATOLOGY.multiyear_averages().

    Calculates the mean over a specified period and groups the data by year-period.
    """

    # Paths to a single or multiple NEMO files.
    fn_nemo_data = '/path/to/nemo/*.nc'
    # Set path for domain file if required.
    fn_nemo_domain = None

    # Read in multiyear data (This example uses NEMO data from multiple datafiles.)
    nemo_data = coast.NEMO(fn_data=fn_nemo_data, fn_domain=fn_nemo_domain, multiple=True, chunks={}).dataset
    # Select specific data variables.
    data = nemo_data[['temperature', 'ssh', 'salinity']]

    # Calculate means of each season across multiple years for specified data.
    clim = coast.CLIMATOLOGY()
    clim_multiyear = clim.multiyear_averages(data, seasons.ALL, time_var='time', time_dim='t_dim')
    return clim_multiyear


