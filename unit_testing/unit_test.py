"""
Script to do unit testing

Written as procedural code that plods through the code snippets and tests the
outputs or expected metadata.

Run:
ipython: cd COAsT; run unit_testing/unit_test.py  # I.e. from the git repo.
"""

import coast
import numpy as np
import xarray as xr


dir = "example_files/"
fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
fn_altimetry = 'COAsT_example_altimetry_data.nc'

sec = 1
subsec = 96 # Code for '`' (1 below 'a')

#################################################
## ( 1 ) Test Loading and initialising methods ##
#################################################

#-----------------------------------------------------------------------------#
# ( 1a ) Load example NEMO data (Temperature, Salinity, SSH)                  #
#                                                                             #
subsec = subsec+1

sci = coast.NEMO() 
sci.load(dir + fn_nemo_dat)


# Test the data has loaded
sci_attrs_ref = dict([('name', 'AMM7_1d_20070101_20070131_25hourm_grid_T'),
             ('description', 'ocean T grid variables, 25h meaned'),
             ('title', 'ocean T grid variables, 25h meaned'),
             ('Conventions', 'CF-1.6'),
             ('timeStamp', '2019-Dec-26 04:35:28 GMT'),
             ('uuid', '96cae459-d3a1-4f4f-b82b-9259179f95f7')])

# checking is LHS is a subset of RHS
if sci_attrs_ref.items() <= sci.dataset.attrs.items(): 
    print(str(sec) + chr(subsec) + " OK - NEMO data loaded: " + fn_nemo_dat)
else:
    print(str(sec) + chr(subsec) + " X - There is an issue with loading " + fn_nemo_dat)

#-----------------------------------------------------------------------------#
# ( 1b ) Load example NEMO domain                                             #
#                                                                             #
subsec = subsec+1

sci_dom = coast.DOMAIN(dir + fn_nemo_dom)

# Test the data has loaded
sci_dom_attrs_ref = dict([('DOMAIN_number_total', 1),
             ('DOMAIN_number', 0),
             ('DOMAIN_dimensions_ids', np.array([1, 2], dtype=np.int32)),
             ('DOMAIN_size_global', np.array([297, 375], dtype=np.int32)),
             ('DOMAIN_size_local', np.array([297, 375], dtype=np.int32)),
             ('DOMAIN_position_first', np.array([1, 1], dtype=np.int32)),
             ('DOMAIN_position_last', np.array([297, 375], dtype=np.int32)),
             ('DOMAIN_halo_size_start', np.array([0, 0], dtype=np.int32)),
             ('DOMAIN_halo_size_end', np.array([0, 0], dtype=np.int32)) ] )

err_flag = False
for key,val in sci_dom_attrs_ref.items():
    # There is somewhere a difference between the arrays
    if (sci_dom.dataset.attrs[key] - val ).any(): 
        print(str(sec) + chr(subsec) + " X - There is an issue with loading " + fn_nemo_dom)
        print( sci_dom.dataset.attrs[key], ': ',val, ': ', 
              (sci_dom.dataset.attrs[key] - val ).any())
        err_flag = True
if err_flag == False:
        print(str(sec) + chr(subsec) + " OK - NEMO domain data loaded: " + fn_nemo_dom)

#-----------------------------------------------------------------------------#
# ( 1c ) Load example altimetry data                                          #
#                                                                             #
subsec = subsec+1

altimetry = coast.ALTIMETRY(dir + fn_altimetry)

# Test the data has loaded using attribute comparison, as for NEMO_data
alt_attrs_ref = dict([('source', 'Jason-1 measurements'),
             ('date_created', '2019-02-20T11:20:56Z'),
             ('institution', 'CLS, CNES'),
             ('Conventions', 'CF-1.6'),])

# checking is LHS is a subset of RHS
if alt_attrs_ref.items() <= altimetry.dataset.attrs.items(): 
    print(str(sec) +chr(subsec) + " OK - Altimetry data loaded: " + fn_altimetry)
else:
    print(str(sec) + chr(subsec) + " X - There is an issue with loading: " + fn_altimetry)

#-----------------------------------------------------------------------------#
# ( 1d ) Load data from existing dataset                                          #
# 
subsec = subsec+1

ds = xr.open_dataset(dir + fn_nemo_dat)
sci_load_ds = coast.NEMO()
sci_load_ds.load_dataset(ds)
sci_load_file = coast.NEMO() 
sci_load_file.load(dir + fn_nemo_dat)
if sci_load_ds.dataset.identical(sci_load_file.dataset):
    print(str(sec) + chr(subsec) + " OK - COAsT.load_dataset()")
else:
    print(str(sec) + chr(subsec) + " X - COAsT.load_dataset() ERROR - not identical to dataset loaded via COAsT.load()")

#################################################
## ( 2 ) Test general utility methods in COAsT ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 2a ) Copying a COAsT object                                               #
#                                                                             #
subsec = subsec+1
altimetry_copy = altimetry.copy()
if altimetry_copy.dataset == altimetry.dataset:
    print(str(sec) +chr(subsec) + " OK - Copied COAsT object ")
else:
    print(str(sec) +chr(subsec) + " X - Copy Failed ")


#################################################
## ( 3 ) Test Transect related methods         ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 3a ) Determining and extracting transect indices                          #
#                                                                             #
subsec = subsec+1

# Extract transect indices
yt, xt, length_of_line = sci_dom.transect_indices([51,-5],[49,-9], grid_ref='t')

# Test transect indices


yt_ref = [164, 163, 162, 161, 161, 160, 159, 158, 157, 156, 155, 155, 154, 153,
          152, 151, 150, 149, 149, 148, 147, 146, 145, 144, 143, 143, 142, 141,
          140, 139, 138, 137, 137, 136, 135, 134]
xt_ref = [134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121,
          120, 119, 118, 117, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106,
          105, 104, 103, 102, 101, 100, 99, 98]
length_ref = 36


if (xt == xt_ref) and (yt == yt_ref) and (length_of_line == length_ref):
    print(str(sec) + chr(subsec) + " OK - NEMO domain transect indices extracted")
else:
    print(str(sec) + chr(subsec) + " X - Issue with indices extraction from NEMO domain transect")


#################################################
## ( 4 ) Object Manipulation (e.g. subsetting) ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 4a ) Subsetting single variable                                           #
#                                                                             #
subsec = subsec+1

# Extact the variable
data_t =  sci.get_subset_as_xarray("votemper", xt_ref, yt_ref)

# Test shape and exteme values

if (np.shape(data_t) == (51, 36)) and (np.nanmin(data_t) - 11.267578 < 1E-6) \
                                  and (np.nanmax(data_t) - 11.834961 < 1E-6):
    print(str(sec) + chr(subsec) + " OK - NEMO COAsT get_subset_as_xarray extracted expected array size and "
          + "extreme values")
else:
    print(str(sec) + chr(subsec) + " X - Issue with NEMO COAsT get_subset_as_xarray method")
    
#-----------------------------------------------------------------------------#
# ( 4b ) Indices by distance method                                           #
#                                                                             #
subsec = subsec+1


# Find indices for points with 111 km from 0E, 51N
ind = sci_dom.subset_indices_by_distance(0,51,111)

# Test size of indices array
if (np.shape(ind) == (2,674)) :
    print(str(sec) + chr(subsec) + " OK - NEMO domain subset_indices_by_distance extracted expected " \
          + "size of indices")
else:
    print(str(sec) + chr(subsec) + "X - Issue with indices extraction from NEMO domain " \
          + "subset_indices_by_distance method")
        
#-----------------------------------------------------------------------------#
# ( 4c ) Subsetting entire COAsT object and return as copy                    #
#                                                                             #
subsec = subsec+1
ind = altimetry.subset_indices_lonlat_box([-10,10], [45,60])
altimetry_nwes = altimetry.isel(time=ind) #nwes = northwest europe shelf

if (altimetry_nwes.dataset.dims['time'] == 213) :
    print(str(sec) + chr(subsec) + " OK - ALTIMETRY object subsetted using isel ")
else:
    print(str(sec) + chr(subsec) + "X - Failed to subset object/ return as copy")

#################################################
## ( 5 ) CRPS Methods                          ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 5a ) Calculate single obs CRPS values                                     #
#                                                                             #
subsec = subsec+1
alt_tmp = altimetry_nwes.subset_as_copy(time=[0,1,2,3,4])
crps_rad = sci.crps_sonf('sossheig', sci_dom, alt_tmp, 'sla_filtered',
                    nh_radius=111, nh_type = "radius", cdf_type = "empirical",
                    time_interp = "nearest", plot=False)

if len(crps_rad)==5:
    print(str(sec) + chr(subsec) + " OK - CRPS SONF done for every observation")
else:
    print(str(sec) + chr(subsec) + " X - Problem with CRPS SONF method")
