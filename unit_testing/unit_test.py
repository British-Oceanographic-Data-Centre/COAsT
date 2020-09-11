"""
Script to do unit testing

Written as procedural code that plods through the code snippets and tests the
outputs or expected metadata.

Run:
ipython: cd COAsT; run unit_testing/unit_test.py  # I.e. from the git repo.
"""

import coast
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime

dn_files = "./example_files/"

if not os.path.isdir(dn_files):
    print(
        "please go download the examples file from https://linkedsystems.uk/erddap/files/COAsT_example_files/")
    dn_files = input("what is the path to the example files:\n")
    if not os.path.isdir(dn_files):
        print(f"location f{dn_files} cannot be found")

dn_fig = 'unit_testing/figures/'
fn_nemo_grid_t_dat_summer = 'nemo_data_T_grid_Aug2015.nc'
fn_nemo_grid_t_dat = 'nemo_data_T_grid.nc'
fn_nemo_grid_u_dat = 'nemo_data_U_grid.nc'
fn_nemo_grid_v_dat = 'nemo_data_V_grid.nc'
fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dat_subset = 'COAsT_example_NEMO_subset_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
fn_altimetry = 'COAsT_example_altimetry_data.nc'
dn_tidegauge = dn_files + 'tide_gauges/'

sec = 1
subsec = 96 # Code for '`' (1 below 'a')

#################################################
## ( 1 ) Test Loading and initialising methods ##
#################################################

#-----------------------------------------------------------------------------#
# ( 1a ) Load example NEMO data (Temperature, Salinity, SSH)                  #
#                                                                             #
subsec = subsec+1

try:
    sci = coast.NEMO(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, grid_ref = 't-grid')

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
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 1b ) Load example altimetry data                                          #
#                                                                             #
subsec = subsec+1

try:
    altimetry = coast.ALTIMETRY(dn_files + fn_altimetry)

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
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 1c ) Load data from existing dataset                                      #
#                                                                             #
subsec = subsec+1
try:
    ds = xr.open_dataset(dn_files + fn_nemo_dat)
    sci_load_ds = coast.NEMO()
    sci_load_ds.load_dataset(ds)
    sci_load_file = coast.NEMO()
    sci_load_file.load(dn_files + fn_nemo_dat)
    if sci_load_ds.dataset.identical(sci_load_file.dataset):
        print(str(sec) + chr(subsec) + " OK - COAsT.load_dataset()")
    else:
        print(str(sec) + chr(subsec) + " X - COAsT.load_dataset() ERROR - not identical to dataset loaded via COAsT.load()")
except:
    print(str(sec) + chr(subsec) +" FAILED")
#-----------------------------------------------------------------------------#
# ( 1d ) Set NEMO variable name                                          #
#
subsec = subsec+1
try:
    sci = coast.NEMO(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, grid_ref='t-grid')
    try:
        sci.dataset.temperature
    except NameError:
        print(str(sec) + chr(subsec) + " X - variable name (to temperature) not reset")
    else:
        print(str(sec) + chr(subsec) + " OK - variable name reset (to temperature)")
except:
    print(str(sec) + chr(subsec) +" FAILED")
#-----------------------------------------------------------------------------#
# ( 1e ) Set NEMO grid attributes - dimension names                                          #
#
subsec = subsec+1
try:
    if sci.dataset.temperature.dims == ('t_dim', 'z_dim', 'y_dim', 'x_dim'):
        print(str(sec) + chr(subsec) + " OK - dimension names reset")
    else:
        print(str(sec) + chr(subsec) + " X - dimension names not reset")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 1f ) Load only domain data in NEMO                #
#                                                                             #
subsec = subsec+1

pass_test = False
nemo_f = coast.NEMO( fn_domain=dn_files+fn_nemo_dom, grid_ref='f-grid' )

if nemo_f.dataset._coord_names == {'depth_0', 'latitude', 'longitude'}:
    var_name_list = []
    for var_name in nemo_f.dataset.data_vars:
        var_name_list.append(var_name)
    if var_name_list == ['e1', 'e2', 'e3_0']:
        pass_test = True

if pass_test:
    print(str(sec) + chr(subsec) + " OK - NEMO loaded domain data only")
else:
    print(str(sec) + chr(subsec) + " X - NEMO didn't load domain data correctly")

#-----------------------------------------------------------------------------#
# ( 1g ) Calculate depth_0 for t,u,v,w,f grids                 #
#                                                                             #
subsec = subsec+1

try:
    nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat,
             fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
    if not np.isclose(np.nansum(nemo_t.dataset.depth_0.values), 1705804300.0):
        raise ValueError(" X - NEMO depth_0 failed on t-grid failed")
    nemo_u = coast.NEMO( fn_data=dn_files+fn_nemo_grid_u_dat,
             fn_domain=dn_files+fn_nemo_dom, grid_ref='u-grid' )
    if not np.isclose(np.nansum(nemo_u.dataset.depth_0.values), 1705317600.0):
        raise ValueError(" X - NEMO depth_0 failed on u-grid failed")
    nemo_v = coast.NEMO( fn_data=dn_files+fn_nemo_grid_v_dat,
             fn_domain=dn_files+fn_nemo_dom, grid_ref='v-grid' )
    if not np.isclose(np.nansum(nemo_v.dataset.depth_0.values), 1705419100.0):
        raise ValueError(" X - NEMO depth_0 failed on v-grid failed")
    nemo_f = coast.NEMO( fn_domain=dn_files+fn_nemo_dom, grid_ref='f-grid' )
    if not np.isclose(np.nansum(nemo_f.dataset.depth_0.values), 1704932600.0):
        raise ValueError(" X - NEMO depth_0 failed on f-grid failed")

    print(str(sec) + chr(subsec) + " OK - NEMO depth_0 calculations correct")
except ValueError as err:
            print(str(sec) + chr(subsec) + str(err))

#-----------------------------------------------------------------------------#
# ( 1h ) Load a subregion dataset with a full domain (AMM7)                #
#                                                                             #
subsec = subsec+1

try:

    amm7 = coast.NEMO(dn_files + fn_nemo_dat_subset,
                     dn_files + fn_nemo_dom)

    # checking all the coordinates mapped correctly to the dataset object
    if amm7.dataset._coord_names == {'depth_0', 'latitude', 'longitude', 'time'}:
        print(str(sec) + chr(subsec) + ' OK - NEMO data subset loaded ', \
              'with correct coords: ' + fn_nemo_dat_subset)
    else:
        print(str(sec) + chr(subsec) + ' X - There is an issue with ', \
              'loading and subsetting the data ' + fn_nemo_dat_subset)

except:
    print(str(sec) + chr(subsec) +' FAILED. Test data in: {}.'\
          .format(fn_nemo_dat_subset) )


#-----------------------------------------------------------------------------#
# ( 1i ) Load and combine (by time) multiple files  (AMM7)               #
#                                                                             #
subsec = subsec+1

try:
    file_names_amm7 = "nemo_data_T_grid*.nc"
    amm7 = coast.NEMO(dn_files + file_names_amm7,
                dn_files + fn_nemo_dom, grid_ref='t-grid', multiple=True)

    # checking all the coordinates mapped correctly to the dataset object
    if amm7.dataset.time.size == 14:
        print(str(sec) + chr(subsec) + ' OK - NEMO data loaded combine ', \
              'over time: ' + file_names_amm7)
    else:
        print(str(sec) + chr(subsec) + ' X - There is an issue with loading',\
              'multiple data files ' + file_names_amm7)

except:
    print(str(sec) + chr(subsec) +' FAILED. Test data in: {} on {}.'\
          .format(dn_files, file_names_amm7) )
        
subsec = subsec+1


#################################################
## ( 2 ) Test general utility methods in COAsT ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 2a ) Copying a COAsT object                                               #
#                                                                             #
subsec = subsec+1

try:
    altimetry_copy = altimetry.copy()
    if altimetry_copy.dataset == altimetry.dataset:
        print(str(sec) +chr(subsec) + " OK - Copied COAsT object ")
    else:
        print(str(sec) +chr(subsec) + " X - Copy Failed ")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 2b ) COAsT __getitem__ returns variable                                   #
#                                                                             #
subsec = subsec+1

try:
    if sci.dataset['sossheig'].equals(sci['sossheig']):
        print(str(sec) +chr(subsec) + " OK - COAsT.__getitem__ works correctly ")
    else:
        print(str(sec) +chr(subsec) + " X - Problem with COAsT.__getitem__ ")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 2c ) Renaming variables inside a COAsT object                             #
#                                                                             #
subsec = subsec+1
try:
    altimetry_copy.rename({'sla_filtered':'renamed'})
    if altimetry['sla_filtered'].equals(altimetry_copy['renamed']):
        print(str(sec) +chr(subsec) + " OK - Renaming of variable in dataset ")
    else:
        print(str(sec) +chr(subsec) + " X - Variable renaming failed ")
except:
    print(str(sec) + chr(subsec) +" FAILED")


#################################################
## ( 3 ) Test Diagnostic methods               ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 3a ) Computing a vertical spatial derivative                              #
#                                                                             #
subsec = subsec+1

# Initialise DataArrays
nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat,
         fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
nemo_w = coast.NEMO( fn_domain=dn_files+fn_nemo_dom, grid_ref='w-grid' )

try:
    log_str = ""
    # Compute dT/dz
    nemo_w_1 = nemo_t.differentiate( 'temperature', dim='z_dim' )
    if nemo_w_1 is None: # Test whether object was returned
        log_str += 'No object returned\n'
    # Make sure the hardwired grid requirements are present
    if not hasattr( nemo_w.dataset, 'depth_0' ):
        log_str += 'Missing depth_0 variable\n'
    if not hasattr( nemo_w.dataset, 'e3_0' ):
        log_str += 'Missing e3_0 variable\n'
    if not hasattr( nemo_w.dataset.depth_0, 'units' ):
        log_str += 'Missing depth units\n'
    # Test attributes of derivative. This are generated last so can indicate earlier problems
    nemo_w_2 = nemo_t.differentiate( 'temperature', dim='z_dim', out_varstr='dTdz', out_obj=nemo_w )
    if not nemo_w_2.dataset.dTdz.attrs == {'units': 'degC/m', 'standard_name': 'dTdz'}:
        log_str += 'Did not write correct attributes\n'
    # Test auto-naming derivative. Again test expected attributes.
    nemo_w_3 = nemo_t.differentiate( 'temperature', dim='z_dim' )
    if not nemo_w_3.dataset.temperature_dz.attrs == {'units': 'degC/m', 'standard_name': 'temperature_dz'}:
        log_str += 'Problem with auto-naming derivative field\n'

    ## Test numerical calculation. Differentiate f(z)=-z --> -1
    # Construct a depth variable - needs to be 4D
    nemo_t.dataset['depth4D'],_ = xr.broadcast( nemo_t.dataset['depth_0'], nemo_t.dataset['temperature'] )
    nemo_w_4 = nemo_t.differentiate( 'depth4D', dim='z_dim', out_varstr='dzdz' )
    if not np.isclose( nemo_w_4.dataset.dzdz.isel(z_dim=slice(1,nemo_w_4.dataset.dzdz.sizes['z_dim'])).max(), -1 ) \
        or not np.isclose( nemo_w_4.dataset.dzdz.isel(z_dim=slice(1,nemo_w_4.dataset.dzdz.sizes['z_dim'])).min(), -1 ):
        log_str += 'Problem with numerical derivative of f(z)=-z\n'

    if log_str == "":
        print(str(sec) + chr(subsec) + " OK - NEMO.differentiate (for d/dz) method passes all tests")
    else:
        print(str(sec) + chr(subsec) + " X - NEMO.differentiate method failed: " + log_str)

except:
    print(str(sec) +chr(subsec) + " X - setting derivative attributes failed ")


#-----------------------------------------------------------------------------#
# ( 3b ) Construct density                                                    #
#                                                                             #
subsec = subsec+1
nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat,
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
nemo_t.construct_density()
yt, xt, length_of_line = nemo_t.transect_indices([54,-15],[56,-12])

try:
    if not np.allclose( nemo_t.dataset.density.sel(x_dim=xr.DataArray(xt,dims=['r_dim']),
                        y_dim=xr.DataArray(yt,dims=['r_dim'])).sum(
                        dim=['t_dim','r_dim','z_dim']).item(),
                        11185010.518671108 ):
        raise ValueError(str(sec) + chr(subsec) + ' X - Density incorrect')
    print(str(sec) + chr(subsec) + ' OK - Density correct')
except ValueError as err:
    print(err)
densitycopy = nemo_t.dataset.density.sel(x_dim=xr.DataArray(xt,dims=['r_dim']),
                        y_dim=xr.DataArray(yt,dims=['r_dim']))

#-----------------------------------------------------------------------------#
# ( 3c ) Construct pycnocline depth and thickness                             #
#                                                                             #
subsec = subsec+1

nemo_t = None; nemo_w = None
nemo_t = coast.NEMO(dn_files + fn_nemo_grid_t_dat_summer,
                    dn_files + fn_nemo_dom, grid_ref='t-grid')
# create an empty w-grid object, to store stratification
nemo_w = coast.NEMO( fn_domain = dn_files + fn_nemo_dom, grid_ref='w-grid')
try:
    log_str = ""
    # initialise Internal Tide object
    IT = coast.INTERNALTIDE(nemo_t, nemo_w)
    if IT is None: # Test whether object was returned
        log_str += 'No object returned\n'
    # Construct pycnocline variables: depth and thickness
    IT.construct_pycnocline_vars( nemo_t, nemo_w )

    if not hasattr( nemo_t.dataset, 'density' ):
        log_str += 'Did not create density variable\n'
    if not hasattr( nemo_w.dataset, 'rho_dz' ):
        log_str += 'Did not create rho_dz variable\n'

    if not hasattr( IT.dataset, 'strat_1st_mom' ):
        log_str += 'Missing strat_1st_mom variable\n'
    if not hasattr( IT.dataset, 'strat_1st_mom_masked' ):
        log_str += 'Missing strat_1st_mom_masked variable\n'
    if not hasattr( IT.dataset, 'strat_2nd_mom' ):
        log_str += 'Missing strat_2nd_mom variable\n'
    if not hasattr( IT.dataset, 'strat_2nd_mom_masked' ):
        log_str += 'Missing strat_2nd_mom_masked variable\n'
    if not hasattr( IT.dataset, 'mask' ):
        log_str += 'Missing mask variable\n'

    # Check the calculations are as expected
    if np.isclose(IT.dataset.strat_1st_mom.sum(), 3.74214231e+08)  \
        and np.isclose(IT.dataset.strat_2nd_mom.sum(), 2.44203298e+08) \
        and np.isclose(IT.dataset.mask.sum(), 450580) \
        and np.isclose(IT.dataset.strat_1st_mom_masked.sum(), 3.71876949e+08) \
        and np.isclose(IT.dataset.strat_2nd_mom_masked.sum(), 2.42926865e+08):
            print(str(sec) + chr(subsec) + " OK - pyncocline depth and thickness good")

except:
    print(str(sec) +chr(subsec) + " X - computing pycnocline depth and thickness failed ")


#-----------------------------------------------------------------------------#
# ( 3d ) Plot pycnocline depth                                                #
#
subsec = subsec+1                                                                             #
try:
    fig,ax = IT.quick_plot( 'strat_1st_mom_masked' )
    fig.tight_layout()
    fig.savefig(dn_fig + 'strat_1st_mom.png')
    print(str(sec) + chr(subsec) + " OK - pycnocline depth plot saved")
except:
    print(str(sec) + chr(subsec) + "X - quickplot() failed")


#################################################
## ( 4 ) Test Transect related methods         ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 4a ) Determining and extracting transect indices                          #
#                                                                             #
subsec = subsec+1

# Extract transect indices
nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat,
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
yt, xt, length_of_line = nemo_t.transect_indices([51,-5],[49,-9])

# Test transect indices
yt_ref = [164, 163, 162, 162, 161, 160, 159, 158, 157, 156, 156, 155, 154,
       153, 152, 152, 151, 150, 149, 148, 147, 146, 146, 145, 144, 143,
       142, 142, 141, 140, 139, 138, 137, 136, 136, 135, 134]
xt_ref = [134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122,
       121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109,
       108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98]
length_ref = 37


if (xt == xt_ref) and (yt == yt_ref) and (length_of_line == length_ref):
    print(str(sec) + chr(subsec) + " OK - NEMO transect indices extracted")
else:
    print(str(sec) + chr(subsec) + " X - Issue with transect indices extraction from NEMO")

#-----------------------------------------------------------------------------#
# ( 4b ) Transport velocity and depth calculations                            #
#
subsec = subsec+1

nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat,
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
nemo_u = coast.NEMO( fn_data=dn_files+fn_nemo_grid_u_dat,
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='u-grid' )
nemo_v = coast.NEMO( fn_data=dn_files+fn_nemo_grid_v_dat,
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='v-grid' )
nemo_f = coast.NEMO( fn_domain=dn_files+fn_nemo_dom, grid_ref='f-grid' )

# Create transect object
tran = coast.Transect( (54,-15), (56,-12), nemo_f, nemo_t, nemo_u, nemo_v )

# Currently we don't have e3u and e3v vaiables so approximate using e3t
e3u = xr.DataArray( tran.data_T.e3t_25h.values,
                   coords={'time': tran.data_U.time},
                   dims=['t_dim', 'z_dim', 'r_dim'])
tran.data_U = tran.data_U.assign(e3=e3u)
e3v = xr.DataArray( tran.data_T.e3t_25h.values,
                   coords={'time': tran.data_U.time},
                   dims=['t_dim', 'z_dim', 'r_dim'])
tran.data_V = tran.data_V.assign(e3=e3v)

output = tran.transport_across_AB()
# Check the calculations are as expected
if np.isclose(tran.data_tran.depth_integrated_transport_across_AB.sum(), -49.19533238588342)  \
        and np.isclose(tran.data_tran.depth_0.sum(), 2301799.05444336) \
        and np.isclose(np.nansum(tran.data_tran.normal_velocities.values), -253.6484375):

    print(str(sec) + chr(subsec) + " OK - TRANSECT transport velocities good")
else:
    print(str(sec) + chr(subsec) + " X - TRANSECT transport velocities not good")

#-----------------------------------------------------------------------------#
# ( 4c ) Transport and velocity plotting                                      #
#
subsec = subsec+1

try:
    plot_dict = {'fig_size':(5,3), 'title':'Normal velocities'}
    fig,ax = tran.plot_normal_velocity(time=0,cmap="seismic",plot_info=plot_dict,smoothing_window=2)
    fig.tight_layout()
    fig.savefig(dn_fig + 'transect_velocities.png')
    plot_dict = {'fig_size':(5,3), 'title':'Transport across AB'}
    fig,ax = tran.plot_depth_integrated_transport(time=0, plot_info=plot_dict, smoothing_window=2)
    fig.tight_layout()
    fig.savefig(dn_fig + 'transect_transport.png')
    print(str(sec) + chr(subsec) + " OK - TRANSECT velocity and transport plots saved")
except:
    print(str(sec) + chr(subsec) + " !!!")

#-----------------------------------------------------------------------------#
# ( 4d ) Construct density on z_levels along transect                         #
#
subsec = subsec+1
tran.construct_density_on_z_levels()
try:
    if not np.allclose( tran.data_T.density_z_levels.sum(dim=['t_dim','r_dim','z_dim']).item(),
                20142532.548826512 ):
        raise ValueError(str(sec) + chr(subsec) + ' X - TRANSECT density on z-levels incorrect')
    # tran.data_T = tran.data_T.drop('density_z_levels')
    # z_levels = tran.data_T.depth_z_levels.copy()
    # tran.data_T = tran.data_T.drop('depth_z_levels')
    # tran.construct_density_on_z_levels( z_levels=z_levels )
    # if not np.allclose( tran.data_T.density_z_levels.sum(dim=['t_dim','r_dim','z_dim']).item(),
    #             20142532.548826512 ):
    #     raise ValueError(str(sec) + chr(subsec) + ' X - TRANSECT density on z-levels incorrect')
    print(str(sec) + chr(subsec) + ' OK - TRANSECT density on z-levels correct')
except ValueError as err:
    print(err)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,4))
densitycopy.isel(t_dim=0).plot.pcolormesh(
            ax=ax1,yincrease=False,y='depth_0')
ax1.set_xticks([0,30])
ax1.set_xticklabels(['A','B'])
tran.data_T.density_z_levels.isel(t_dim=0).plot.pcolormesh(
    ax=ax2,yincrease=False, y='depth_z_levels')
plt.xticks([0,57],['A','B'])
plt.show()

#################################################
## ( 5 ) Object Manipulation (e.g. subsetting) ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 5a ) Subsetting single variable                                           #
#                                                                             #
subsec = subsec+1

try:
    # Extact the variable
    data_t =  sci.get_subset_as_xarray("temperature", xt_ref, yt_ref)

    # Test shape and exteme values
    if (np.shape(data_t) == (51, 37)) and (np.nanmin(data_t) - 11.267578 < 1E-6) \
                                      and (np.nanmax(data_t) - 11.834961 < 1E-6):
        print(str(sec) + chr(subsec) + " OK - NEMO COAsT get_subset_as_xarray extracted expected array size and "
              + "extreme values")
    else:
        print(str(sec) + chr(subsec) + " X - Issue with NEMO COAsT get_subset_as_xarray method")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 5b ) Indices by distance method                                           #
#                                                                             #
subsec = subsec+1

try:
    # Find indices for points with 111 km from 0E, 51N

    ind = sci.subset_indices_by_distance(0,51,111)

    # Test size of indices array
    if (np.shape(ind) == (2,674)) :
        print(str(sec) + chr(subsec) + " OK - NEMO domain subset_indices_by_distance extracted expected " \
              + "size of indices")
    else:

        print(str(sec) + chr(subsec) + "X - Issue with indices extraction from NEMO domain " \
              + "subset_indices_by_distance method")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 5c ) Subsetting entire COAsT object and return as copy                    #
#                                                                             #
subsec = subsec+1
try:
    ind = altimetry.subset_indices_lonlat_box([-10,10], [45,60])
    altimetry_nwes = altimetry.isel(t_dim=ind) #nwes = northwest europe shelf

    if (altimetry_nwes.dataset.dims['t_dim'] == 213) :
        print(str(sec) + chr(subsec) + " OK - ALTIMETRY object subsetted using isel ")
    else:
        print(str(sec) + chr(subsec) + "X - Failed to subset object/ return as copy")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 5d ) Find nearest xy indices                                              #
#                                                                             #
subsec = subsec+1
try:
    ind_x, ind_y = sci.nearest_xy_indices(sci.dataset,
                                          altimetry_nwes.dataset.longitude,
                                          altimetry_nwes.dataset.latitude)
    if ind_x.shape == altimetry_nwes.dataset.longitude.shape:
        print(str(sec) + chr(subsec) + " OK - nearest_xy_indices works ")
    else:
        print(str(sec) + chr(subsec) + "X - Problem with nearest_xy_indices()")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 5e ) Interpolate in space (nearest)                                       #
#                                                                             #
subsec = subsec+1
try:
    interp_lon = np.array(altimetry_nwes.dataset.longitude).flatten()
    interp_lat = np.array(altimetry_nwes.dataset.latitude).flatten()
    interpolated = sci.interpolate_in_space(sci.dataset.sossheig,
                                            interp_lon, interp_lat)

    # Check that output array longitude has same shape as altimetry
    if interpolated.longitude.shape == altimetry_nwes.dataset.longitude.shape :
        print(str(sec) + chr(subsec) + " OK - Space interpolation works ")
    else:
        print(str(sec) + chr(subsec) + "X - Problem with space interpolation")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 5f ) Interpolate in time                                                  #
#                                                                             #
subsec = subsec+1
try:
    interpolated = sci.interpolate_in_time(interpolated,
                                           altimetry_nwes.dataset.time)

    #Check time in interpolated object has same shape
    if interpolated.time.shape == altimetry_nwes.dataset.time.shape :
        print(str(sec) + chr(subsec) + " OK - ALTIMETRY object subsetted using isel ")
    else:
        print(str(sec) + chr(subsec) + "X - Failed to subset object/ return as copy")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#################################################
## ( 6 ) Validation Methods                    ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 6a ) Calculate single obs CRPS values                                     #
#                                                                             #
subsec = subsec+1
try:
    nemo = coast.NEMO(dn_files + fn_nemo_dat, dn_files + fn_nemo_dom, grid_ref = 't-grid')
    altimetry = coast.ALTIMETRY(dn_files + fn_altimetry)
    ind = altimetry.subset_indices_lonlat_box([-10,10], [45,60])
    altimetry_nwes = altimetry.isel(t_dim=ind) #nwes = northwest europe shelf
    crps = coast.CRPS(nemo, altimetry_nwes, 'sossheig','sla_filtered', nh_radius=30)

    try:
        if len(crps.dataset.crps)==len(altimetry_nwes['sla_filtered']):
            print(str(sec) + chr(subsec) + " OK - CRPS SONF done for every observation")
        else:
            print(str(sec) + chr(subsec) + " X - Problem with CRPS SONF method")

            if len(crps.crps)==len(altimetry_nwes['sla_filtered']):
                print(str(sec) + chr(subsec) + " OK - CRPS SONF done for every observation")
            else:
                print(str(sec) + chr(subsec) + " X - Problem with CRPS SONF method")
    except:
        print(str(sec) + chr(subsec) +" FAILED")
except:
    print(str(sec) + chr(subsec) +" FAILED")

#-----------------------------------------------------------------------------#
# ( 6b ) CRPS Map Plots                                                       #
#                                                                             #
subsec = subsec+1
plt.close('all')
try:
    fig, ax = crps.map_plot()
    fig.savefig(dn_fig + 'crps_map_plot.png')
    #plt.close(fig)
    print(str(sec) + chr(subsec) + " OK - CRPS Map plot saved")
except:
    print(str(sec) + chr(subsec) + " X - CRPS Map plot not saved")

#-----------------------------------------------------------------------------#
# ( 6c ) CRPS Map Plots                                                       #
#                                                                             #

plt.close('all')
subsec = subsec+1
try:
    fig, ax = crps.cdf_plot(0)
    fig.savefig(dn_fig + 'crps_cdf_plot.png')
    #plt.close(fig)
    print(str(sec) + chr(subsec) + " OK - CRPS CDF plot saved")
except:
    print(str(sec) + chr(subsec) + " X - CRPS CDF plot not saved")

#-----------------------------------------------------------------------------#
# ( 6d ) Interpolate model to altimetry                                       #
#                                                                             #
subsec = subsec+1
plt.close('all')

try:
    altimetry_nwes.obs_operator(sci, 'sossheig')
    # Check new variable is in altimetry dataset and isn't all NaNs
    try:
        test = altimetry_nwes.dataset.interp_sossheig
        if False in np.isnan(altimetry_nwes.dataset.interp_sossheig):
            print(str(sec) + chr(subsec) + " OK - SSH interpolated to altimetry")
        else:
            print(str(sec) + chr(subsec) + " OK - X - Interpolation to altimetry failed")
    except:
        print(str(sec) + chr(subsec) + " X - Interpolation to altimetry failed")
except:
    print(str(sec) + chr(subsec) + " FAILED")


#################################################
## ( 7 ) Plotting Methods                      ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 7a ) Altimetry quick_plot()                                               #
#                                                                             #
subsec = subsec+1
plt.close('all')

try:
    fig, ax = altimetry.quick_plot('sla_filtered')
    fig.savefig(dn_fig + 'altimetry_quick_plot.png')
    #plt.close(fig)
    print(str(sec) + chr(subsec) + " OK - Altimetry quick plot saved")
except:
    print(str(sec) + chr(subsec) + " X - Altimetry quick plot not saved")

plt.close('all')

#################################################
## ( 8 ) TIDEGAUGE Methods                     ##
#################################################
sec = sec+1
subsec = 96

#-----------------------------------------------------------------------------#
# ( 8a ) Load in GESLA tide gauge files from directory                        #
#                                                                             #
subsec = subsec+1

try:
    date0 = datetime.datetime(2010,1,1)
    date1 = datetime.datetime(2010,12,1)
    tg = coast.TIDEGAUGE(dn_tidegauge, date_start = date0, date_end = date1)
    
    # Check length of dataset_list is correct and that
    test_attrs = {'site_name': 'FUKAURA', 'country': 'Japan',
    'contributor': 'Japan_Meteorological_Agency',
    'latitude': 40.65, 'longitude': 139.9333,
    'coordinate_system': 'Unspecified',
    'original_start_date': np.datetime64('1971-12-31 15:00:00'),
    'original_end_date': np.datetime64('2013-12-31 14:00:00'),
    'time_zone_hours': 0.0, 'precision': 0.01, 'null_value': -99.9999}
    if len(tg.dataset_list) == 9 and tg.dataset_list[0].attrs == test_attrs:
        print(str(sec) + chr(subsec) + " OK - Tide gauges loaded")
except:
    print(str(sec) + chr(subsec) +' FAILED.')

#-----------------------------------------------------------------------------#
# ( 8b ) TIDEGAUGE map plot                                                   #
#                                                                             #
subsec = subsec+1

try:
    f,a = tg.plot_map()
    f.savefig(dn_fig + 'tidegauge_map.png')
    print(str(sec) + chr(subsec) + " OK - Tide gauge map plot saved")
except:
    print(str(sec) + chr(subsec) +' FAILED.')
    
plt.close('all')

#-----------------------------------------------------------------------------#
# ( 8c ) TIDEGAUGE Time series plot                                           #
#                                                                             #
subsec = subsec+1


try:
    f,a = tg.plot_timeseries(0)
    f.savefig(dn_fig + 'tidegauge_timeseries.png')
    print(str(sec) + chr(subsec) + " OK - Tide gauge time series saved")
except:
    print(str(sec) + chr(subsec) +' FAILED.')
    
plt.close('all')
