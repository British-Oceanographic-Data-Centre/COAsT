"""
Script to do unit testing

Written as procedural code that plods through the code snippets and tests the
outputs or expected metadata.

Run:
ipython: cd COAsT; run unit_testing/unit_test.py  # I.e. from the git repo.
"""


import coast
import numpy as np


dir = "example_files/"

## Test COAsT class methods
###########################

# Load example data (Temperature)
sci = coast.NEMO()
sci.load(dir + "COAsT_example_data_141020.nc")


# Test the data has loaded
sci_attrs_ref = dict([('name', 'AMM7_1d_20070101_20070131_25hourm_grid_T'),
             ('description', 'ocean T grid variables, 25h meaned'),
             ('title', 'ocean T grid variables, 25h meaned'),
             ('Conventions', 'CF-1.6'),
             ('timeStamp', '2019-Dec-26 04:35:28 GMT'),
             ('uuid', '96cae459-d3a1-4f4f-b82b-9259179f95f7')])

if sci_attrs_ref.items() <= sci.dataset.attrs.items(): # checking is LHS is a subset of RHS
    print("OK - NEMO data loaded: COAsT_example_data_141020.nc")
else:
    print("X - There is an issue with loading COAsT_example_data_141020.nc")


# Load example domain file
sci_dom = coast.DOMAIN()
sci_dom.load(dir + "COAsT_example_domain_141020.nc")

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
    if (sci_dom.dataset.attrs[key] - val ).any(): # There is somewhere a difference between the arrays
        print("X - There is an issue with loading COAsT_example_domain_141020.nc")
        print( sci_dom.dataset.attrs[key], ': ',val, ': ', (sci_dom.dataset.attrs[key] - val ).any())
        err_flag = True
if err_flag == False:
        print("OK - NEMO domain data loaded: COAsT_example_domain_141020.nc")


## Test DOMAIN class methods
############################

## Test transect

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
    print("OK - NEMO domain transect indices extracted")
else:
    print("X - Issue with indices extraction from NEMO domain transect")



## Test distance from point method

# Find indices for points with 111 km from 0E, 51N
ind = sci_dom.subset_indices_by_distance(0,51,111)

# Test size of indices array
if (np.shape(ind) == (2,674)) :
    print("OK - NEMO domain subset_indices_by_distance extracted expected " \
          + "size of indices")
else:
    print("X - Issue with indices extraction from NEMO domain " \
          + "subset_indices_by_distance method")


## Test more COAsT class methods
################################

## Test variable extract along transect: get_subset_as_xarray

# Extact the variable
data_t =  sci.get_subset_as_xarray("votemper", xt_ref, yt_ref)

# Test shape and exteme values

if (np.shape(data_t) == (51, 36)) and (np.nanmin(data_t) - 11.267578 < 1E-6) \
                                  and (np.nanmax(data_t) - 11.834961 < 1E-6):
    print("OK - NEMO COAsT get_subset_as_xarray extracted expected array size and "
          + "extreme values")
else:
    print("X - Issue with NEMO COAsT get_subset_as_xarray method")



## Test ALTIMETRY class methods
###############################

"""    PENDING GETTING INTO develop
dir2 = "/Users/jeff/Downloads/"

fn_dom = dir2 + "COAsT_example_NEMO_domain.nc"
fn_dat = dir2 + "COAsT_example_NEMO_data.nc"
fn_alt = dir2 + "COAsT_example_altimetry_data.nc"

nemo_dom = coast.DOMAIN()
nemo_var = coast.NEMO()
alt_test = coast.ALTIMETRY()

nemo_dom.load(fn_dom)
nemo_var.load(fn_dat)
alt_test.load(fn_alt)

alt_test.set_command_variables()
nemo_var.set_command_variables()
nemo_dom.set_command_variables()

# Extract lon/lat box (saves into alt_test object)
alt_test.extract_lonlat_box([-10,10], [45,65])
# Just use the first 3 elements of remaining altimetry data
alt_test.extract_indices_all_var(np.arange(0,4))

crps_test = nemo_var.crps_sonf('ssh', nemo_dom, alt_test, 'sla_filtered',
                    nh_radius=111, nh_type = "radius", cdf_type = "empirical",
                    time_interp = "nearest", plot=True)

"""
