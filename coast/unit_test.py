"""
Script to do unit testing

Written as procedural code that plods through the code snippets and tests the
outputs or expected metadata.

Run:
ipython: cd COAsT; run coast/unit_test.py  # I.e. from the git repo.
"""


import coast
import numpy as np


dir = "example_files/"


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
    print("NEMO data loaded: COAsT_example_data_141020.nc")
else:
    print("There is an issue with loading COAsT_example_data_141020.nc")



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
        print("There is an issue with loading COAsT_example_domain_141020.nc")
        print( sci_dom.dataset.attrs[key], ': ',val, ': ', (sci_dom.dataset.attrs[key] - val ).any())
        err_flag = True
if err_flag == False:
        print("NEMO domain data loaded: COAsT_example_domain_141020.nc")




## Test transect

# Extract transect indices
yt, xt, length_of_line = sci_dom.transect_indices([42,-3],[43,-2], grid_ref='t')

# Test transect indices
yt_ref = [29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44]
xt_ref = [152, 153, 153, 154, 155, 155, 156, 156, 157, 158, 158, 159, 160, 160, 161]
length_ref = 15


if (xt == xt_ref) and (yt == yt_ref) and (length_of_line == length_ref):
    print("NEMO domain transect indices extracted")
else:
    print("Issue with indices extraction from NEMO domain transect")



## Test distance from point method

# Find indices for points with 111 km from 0E, 51N
ind = sci_dom.subset_indices_by_distance(0,51,111)

# Test size of indices array
if (np.shape(ind) == (2,674)) :
    print("NEMO domain subset_indices_by_distance extracted expected size of indices")
else:
    print("Issue with indices extraction from NEMO domain subset_indices_by_distance method")
