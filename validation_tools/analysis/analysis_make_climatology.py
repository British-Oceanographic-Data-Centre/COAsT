"""
@author: David Byrne (dbyrne@noc.ac.uk)
v1.0 (20-01-2021)

This script uses COAsT and xarray to calculate a climatological mean of an
input dataset at a desired output frequency. Output will be written straight
to file. 

There are two example routines for reading data into the correct input format.
These are for NEMO output data and OSTIA SST data. If the output of a read
routine is correct, the rest of the script should work. See COAsT.CLIMATOLOGY()
for more information.

COAsT and xarray should preserve any lazy loading and chunking. If defined
properly in the read function, memory issues can be avoided and parallel
processes will automatically be used.

The script is in three parts: global variables, functions and the main script.
It is modular and designed for functions (especially reading) to be swapped
out.

*NOTE: In all xarray.open_dataset or xarray.open_mfdataset calls, make sure
you switch on Dask by defining chunks. At the least, pass the argument
chunks = {} OR chunks = 'auto'.
"""
# Import necessary packages

# UNCOMMENT IF USING DEVELOPMENT VERSION OF COAsT (git clone)
#import sys
#sys.path.append('<PATH TO COAsT DIRECTORY>')

import coast
import xarray as xr

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # MAIN SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def main():
    
    # SET VARIABLES #######################################
    
    # Paths to NEMO files if using
    fn_nemo_data = '<FULL PATH TO NEMO DATA FILE(s)>'
    fn_nemo_domain = '<FULL PATH TO NEMO DOMAIN FILE>'

    # Define output file
    fn_out = "<FULL PATH TO DESIRED OUTPUT CLIMATOLOGY FILE>"
    
    # Define frequency -- Any xarray time string: season, month, etc
    climatology_frequency = 'season'
    #######################################################
    
    # Use a READ routine to create an xarray dataset
    data = read_data_input_nemo(fn_nemo_data, fn_nemo_domain)
    
    # Calculate the climatology and write to file.
    calculate_climatology_using_coast(data, climatology_frequency, fn_out)

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # FUNCTIONS
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def read_data_input_ostia(fn_nemo_ostia):
    ''' For reading multiple OSTIA input files to use for analysis '''
    
    fn_ostia = "/Users/dbyrne/Projects/CO9_AMM15/data/ostia/*.nc"
    
    kelvin_to_celcius = -273.15
    data = xr.open_mfdataset(fn_ostia, chunks='auto', concat_dim='time', 
                          parallel=True)
    data = data.rename({'analysed_sst':'temperature'})
    data = data.rename({'time':'t_dim'})
    data['temperature'] = data.temperature + kelvin_to_celcius
    data.attrs = {}
    
    return data

def read_data_input_nemo(fn_nemo_data, fn_nemo_domain):
    ''' For reading multiple NEMO data files to use for analysis. Uses COAsT
    to create the xarray dataset.'''
    data = coast.NEMO(fn_nemo_data, fn_nemo_domain, multiple=True,
                      chunks='auto').dataset
    data = data[['temperature','ssh','salinity']]
    return data

def calculate_climatology_using_coast(data, climatology_frequency, fn_out):
    # COAsT climatology
    CLIM = coast.CLIMATOLOGY()
    clim_mean = CLIM.make_climatology(data, climatology_frequency, 
                                      fn_out=fn_out)
    return clim_mean

