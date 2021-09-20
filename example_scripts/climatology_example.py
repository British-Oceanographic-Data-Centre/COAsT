"""
This file has two main example scripts:

1)  make_climatology():
    This script uses the COAsT package to calculate a climatological mean of an
    input dataset at a desired output frequency. Output will be written straight
    to file.

2) make_multiyear_climatology():
    This script uses the COAsT package to calculate a climatological mean of an
    input dataset at a desired output frequency, over multiple years.

COAsT and xarray should preserve any lazy loading and chunking. If defined
properly in the read function, memory issues can be avoided and parallel
processes will automatically be used.
"""

import coast
from coast import seasons


def make_climatology():
    """Example code explaining the usage of coast.CLIMATOLOGY.make_climatology().

    Calculates mean over a given period of time. This doesn't take different years into account, unless using the
    'years' frequency.
    """

    # Paths to a single or multiple data files.
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


def make_multiyear_climatology():
    """Example code explaining the usage of coast.CLIMATOLOGY.multiyear_averages().

    Calculates the mean over a specified period and groups the data by year-period.
    """

    # Paths to a single or multiple data files.
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
