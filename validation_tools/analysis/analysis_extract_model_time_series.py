#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 1.0
Date: 25/02/2021
@author: David Byrne (dbyrne@noc.ac.uk)

This script with uses COAsT to extract the nearest model locations to 
a list of longitude/latitude pairs and saves the resulting set of time series 
to file. The script is modular in nature, with 'read' functions, functions
for defining latitude/longitudes and functions for doing the extraction. By 
default, the script is set up to read NEMO data (using COAsT) in an xarray
dataset and extract locations around the UK (Liverpool and Southampton).
CHANGE the contents of define_locations_to_extract() to set extraction
longitude/latitudes.

Any functions can be changed, and as long as the correct data format is 
adhered to, the rest of the script should continue to work. Model data
should be read into an xarray.dataset object with COAsT dimension and 
coordinate names (dims = (x_dim, y_dim, z_dim, t_dim), coords = (time,
latitude, longitude, depth)). Longitudes and latitudes to extract should be
provided as 1D numpy arrays.

The saved timeseries file can be opened using xarray:
    timeseries = xr.open_dataset(file_timeseries, chunks={})
Inspecting this object will reveal a new 'location' dimension, which is 
the locations in order of input into the script. There will also be the time
dimension (t_dim) and if the input data had depth, then z_dim will be retained.
"""
# Import necessary packages

# UNCOMMENT IF USING DEVELOPMENT VERSION OF COAsT (git clone)
#import sys
#sys.path.append('<PATH TO COAsT DIRECTORY>')

import coast
import coast.general_utils as general_utils
import numpy as np
import xarray as xr
import os

'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # MAIN SCRIPT
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def main():
    # SET VARIABLES #########################################################
    # NEMO data and domain files if using read_model_nemo()
    fn_nemo_data = '<PATH TO NEMO DATA FILE(S)>'
    fn_nemo_domain = '<PATH TO NEMO DOMAIN FILE>'

    # Output file to save timeseries -- any existing files will be deleted.
    fn_timeseries = "<PATH TO DESIRED OUTPUT FILE>"
    
    # Which depth levels to extract. 0 = surface. set to 'all' to extract
    # all depths. Alternatively, set to an array of depth levels.
    # If file has no depth dimension, will do nothing.
    depth_levels = 0
    
    # Which variables to extract -- a list of strings or 'all'.
    variables_to_extract = 'all'
    #########################################################################
    
    # Read or create new longitude/latitudes.
    extract_lon, extract_lat = define_locations_to_extract()
    
    # Read data to extract from
    model_data = read_model_nemo(fn_nemo_data, fn_nemo_domain, 
                            depth_levels, variables_to_extract)
    
    # Extract model locations nearest to extract_lon and extract_lat
    indexed = extract_nearest_points_using_coast(model_data, extract_lon, 
                                                 extract_lat)
    
    # Write indexed dataset to file
    write_timeseries_to_file(indexed, fn_timeseries)


'''
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 # FUNCTIONS
 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''

def define_locations_to_extract():
    ''' Routine for defining the lat/lon locations to extract from model data.
    This should return numpy arrays of longitude and latitude. This can be done
    manually or by reading data from another file and extracting lists of 
    locations. '''
    
    # Liverpool and Southampton
    extract_lon = np.array( [-3.018, -1.3833] )
    extract_lat = np.array( [53.45, 50.9 ] )
    
    return extract_lon, extract_lat

def read_model_nemo(fn_nemo_data, fn_nemo_domain, 
                    depth_levels='all', variables_to_extract='all'):
    ''' Routine for reading NEMO model data using COAsT.
    This should return numpy arrays of longitude and latitude. This can be done
    manually or by reading data from another file and extracting lists of 
    locations'''
    
    # Read NEMO data into a COAsT object (correct format)
    model_data = coast.NEMO(fn_nemo_data, fn_nemo_domain, grid_ref = 't-grid', 
                       multiple=True, chunks={'time_counter':1})
    
    # Extract the xarray dataset and desired variables
    model_data = model_data.dataset
    
    # If dataset has a depth dimension and user has specified depth levels
    if depth_levels != 'all' and 'z_dim' in model_data.dims:
        model_data = model_data.isel(z_dim=depth_levels)
    
    # If only specified variables are wanted
    if variables_to_extract != 'all':
        model_data = model_data[variables_to_extract]
        
    # Create a landmask and place into dataset
    # Here I create a landmask from the top_level variable in the domain file.
    # This should be named 'landmask'.
    domain = xr.open_dataset(fn_nemo_domain, chunks = {})
    model_data['landmask'] = (['y_dim','x_dim'],~domain.top_level[0].values.astype(bool))
    
    # If no mask needed, set to None (uncomment below)
    # model_data['landmask'] = None
    return model_data

def extract_nearest_points_using_coast(model_data, extract_lon, extract_lat):
    '''
    Use COAsT to identify nearest model points and extract them into a new
    xarray dataset, ready for writing to file or using directly.
    '''
    # Use COAsT general_utils.nearest_indices_2D routine to work out the model
    # indices we want to extract
    ind2D = general_utils.nearest_indices_2D(model_data.longitude, model_data.latitude,
                                             extract_lon, extract_lat,
                                             mask = model_data.landmask)
    print('Calculated nearest model indices using BallTree.')

    # Extract indices into new array called 'indexed'
    indexed = model_data.isel(x_dim = ind2D[0], y_dim = ind2D[1])
    
    # Determine distances from extracted locations and save to dataset.
    # Can be used to check points outside of domain or similar problems.
    indexed_dist = general_utils.calculate_haversine_distance(extract_lon, 
                                                          extract_lat, 
                                                          indexed.longitude.values,
                                                          indexed.latitude.values)
    
    # If there is more than one extract location, 'dim_0' will be a dimension
    # in indexed.
    if 'dim_0' in indexed.dims:
        # Rename the index dimension to 'location'
        indexed = indexed.rename({'dim_0':'location'})
        indexed['dist_from_nearest_neighbour'] = ('location', indexed_dist)
    else:
        indexed['dist_from_nearest_neighbour'] = indexed_dist
        
    indexed['model_indices_x'] = ('location', ind2D[0])
    indexed['model_indixes_y'] = ('location', ind2D[1])
        
    return indexed

def write_timeseries_to_file(indexed, fn_timeseries):
    ''' Write extracted data to file '''
    if os.path.exists(fn_timeseries):
        os.remove(fn_timeseries)
    print('Writing to file. For large datasets over multiple files, this may take some time')
    indexed.to_netcdf(fn_timeseries)
    
if __name__ == '__main__':
    main()
