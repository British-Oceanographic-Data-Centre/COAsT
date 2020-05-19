import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ds
import datetime
import coast
import xarray as xr

fn_dom = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_NEMO_domain.nc'
fn_dat = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_NEMO_data.nc'
fn_alt = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_altimetry_data.nc'

nemo_dom = coast.DOMAIN()
nemo_var = coast.NEMO()

nemo_dom.load(fn_dom)
nemo_var.load(fn_dat)

co = coast.COAsT()

# altimetry read
ncalt = ds(fn_alt)
alt_lon = ncalt.variables['longitude'][:]
alt_lat = ncalt.variables['latitude'][:]
alt_ssh = ncalt.variables['sla_filtered'][:]
alt_time = ncalt.variables['time'][:]
ncalt.close()

alt_epoch = datetime.datetime(1950,1,1)
alt_units = 'days'
alt_time = alt_time
alt_time = co.num_to_date(alt_time, alt_epoch, alt_units)

alt_lon[alt_lon>180] = alt_lon[alt_lon>180] - 360

ind_box = co.extract_lonlat_box(alt_lon, alt_lat, [-10,10],[45,65])
alt_lon = xr.DataArray( alt_lon[ind_box] )
alt_lat = xr.DataArray( alt_lat[ind_box] )
alt_time = xr.DataArray( alt_time[ind_box] )
alt_ssh = xr.DataArray( alt_ssh[ind_box] )

# Which observation do we actually want to use for CRPS
# Determine best model time to use.
crps_test = nemo_var.crps_sonf("sossheig", nemo_dom,
                       alt_lon[0], alt_lat[0], alt_ssh[0], alt_time[0],
                       plot=True, cdf_type = 'empirical', nh_type = 'radius')




