"""
Created on Fri Jun 10 11:53:51 2022

transform/strech from 360 (used often in climate models to 365-366 days
!!Attention when the data are higher frequency that daily e.g. 3-hourly
as the strech/interpolation has to preserve the daily cycle                    
This is an example for near surface air temperature from CMIP6-UKESM model 
during the historical period
Method 1 and Method 2 can both be used and depend on the application:
Method 1: streches 
Method 2: add extra days through the year
                    
@author: annkat
"""
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mc  # fancy symetric colours on log scale
import matplotlib.colors as colors  # colormap fiddling
import calendar
from scipy.interpolate import interp1d
from sklearn import datasets
import xarray as xr

# Method 1 generalised:
# TODO: get to work with multiyear data?
def stretch_time(time_var: np.ndarray, data_var: np.ndarray, year: int = 1850, hourly_interval: int = 24):
    """Returns a copy of the time and interpolated data variable where time is stretched to 365/366 days.
    
    Args:
        time_var (ndarray): Time variable.
        data_var (ndarray): Data variable.
        year (int): Year of the data.
        hourly_interval (int): Interval of data. Defaults to 24 for daily. 
    Returns:
        Tuple[ndarray, ndarray]: Tuple containing the new time and data arrays. (Time, Data).
    
    """

    if calendar.isleap(year):
        days_in_year = 366
    else:
        days_in_year = 365

    days_in_data = time_var.size / (24 / hourly_interval)
    time_original = np.arange(1, days_in_data + 1, 1)
    count = days_in_data / days_in_year
    time_stretch_method = np.arange(count, days_in_data + count, count)

    measures_per_day = int(24 / hourly_interval)  # number of measurments, every hourly interval
    interpolated_data = np.empty(shape=(days_in_year * measures_per_day))
    for ih in range(0, measures_per_day):
        tt = np.interp(
            time_stretch_method, time_original, np.squeeze(data_var[ih : time_var.size : measures_per_day, 0, 0])
        )
        interpolated_data[ih : days_in_year * measures_per_day : measures_per_day] = tt
    count = 1 / (measures_per_day)
    new_time = np.arange(count, days_in_year + count, count)
    return new_time, interpolated_data



def old_add_time(dataset: Dataset, time_var_name: str = "time", year: int = 1850, hourly_interval: int = 24):
    """Add days to allow add leap year support."""
    
    time_var = dataset.variables[time_var_name][:]
    
    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    
    days_in_data = int(time_var.size / (24 / hourly_interval))
    
    # first and last 15 stay the same
    time_str_method2 = np.arange(1, 15 + 1, 1)
    count_n = int((days - 15 * 2) / (days - days_in_data))
    extra_days = days - days_in_data
    t_end = time_str_method2[-1]
    for id in range(1, extra_days + 1):
        tt = np.arange(t_end + 1, count_n + t_end, 1)
        time_str_method2 = np.append(time_str_method2, tt)
        t_end = time_str_method2[-1]
        time_str_method2 = np.append(time_str_method2, (t_end + 0.5))

    # add the last 15 days
    tt = np.arange(time_str_method2[-1] + 0.5, days_in_data + 1, 1)
    time_str_method2 = np.append(time_str_method2, tt)

    # for hourly fields
    n_m = int(24 / hourly_interval)  # number of measurments, every 3 hours
    count = 1 / (n_m)
    time_tas_365 = np.arange(count, days + count, count)
    
    # Create new stretched time dim and variable.
    dataset.createDimension("stretched_time", size = time_tas_365.size)
    dataset.createVariable("stretched_time", float, dimensions=("stretched_time",) )
    dataset.variables["stretched_time"][:] = time_tas_365
    
    dataset_variables = [var for _,var in dataset.variables.items()]
    for data_var in dataset_variables:
        if time_var_name not in data_var.dimensions:
            continue
        try:
            tas_str_method2 = np.empty(shape=(days * n_m))
            for ih in range(0, n_m):
                tt = np.interp(time_str_method2, time_or, np.squeeze(data_var[ih : time_var.size : n_m, 0, 0]))
                tas_str_method2[ih : days * n_m : n_m] = tt
            # Create new stretched variable. 
            new_var_name = f"{data_var.name}_stretched"
            dataset.createVariable(new_var_name, float, dimensions=("stretched_time",))
            dataset.variables[new_var_name][:] = tas_str_method2
        except Exception as exc:
            print(exc)
        
    return dataset
    
    
    
def add_time(dataset: xr.Dataset, time_var_name: str = "time", year: int = 1850, hourly_interval: int = 24):
    """Add days to allow add leap year support."""
    
    time_var = dataset[time_var_name][:]
    
    if calendar.isleap(year):
        days = 366
    else:
        days = 368
    
    days_in_data = int(time_var.size / (24 / hourly_interval))
    time_or = np.arange(1, days_in_data + 1, 1)
    
    # first and last 15 stay the same
    time_str_method2 = np.arange(1, 15 + 1, 1)
    count_n = int((days - 15 * 2) / (days - days_in_data))
    extra_days = days - days_in_data
    t_end = time_str_method2[-1]
    for _ in range(1, extra_days + 1):
        tt = np.arange(t_end + 1, count_n + t_end, 1)
        time_str_method2 = np.append(time_str_method2, tt)
        t_end = time_str_method2[-1]
        time_str_method2 = np.append(time_str_method2, (t_end + 0.5))

    # add the last 15 days
    tt = np.arange(time_str_method2[-1] + 0.5, days_in_data + 1, 1)
    time_str_method2 = np.append(time_str_method2, tt)

    # for hourly fields
    n_m = int(24 / hourly_interval)  # number of measurments, every 3 hours
    count = 1 / (n_m)
    time_tas_365 = np.arange(count, days + count, count)
    
    stretched_variables = []
    for var_name, data_var in dataset.variables.items():
        if time_var_name not in data_var.dims:
            continue
        try:
            tas_str_method2 = np.empty(shape=(days * n_m))
            for ih in range(0, n_m):
                #tt = np.interp(time_str_method2, time_or, np.squeeze(data_var[ih : time_var.size : n_m, 0, 0]))
                interpolate2d = interp1d(x = time_or, y=data_var[:], axis=0, fill_value="extrapolate")
                new_data = interpolate2d(time_str_method2)
                #tas_str_method2[ih : days * n_m : n_m] = tt
            # Create new stretched variable. 
            dim_dict = {dim: dataset[dim][:] for dim in data_var.dims if dim != time_var_name}
            dim_dict['time'] = time_tas_365
            # Create new data array for strected variable.
            data_array = xr.DataArray(data=new_data, coords=dim_dict, name=var_name, dims=("time", "i", "j"))
            stretched_variables.append(data_array)
        except Exception as exc:
            print(exc)
        # Create new dataset from all stretched variables.
        new_dataset = xr.Dataset(data_vars={var.name: var for var in stretched_variables})
    return new_dataset
    
    
    
    
    



data_dir = Path("E:\\COAsT data\\")
three_hour = data_dir / "tas_3hr_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"
daily = data_dir / "tos_Oday_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"
three_hour_edit = data_dir / "tas_3hr_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset_edit.nc"
daily_edit = data_dir / "tos_Oday_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset_edit.nc"
daily_dataset = xr.load_dataset(daily)

#########################
# Read in file
# this is a 1 year files: for 3 hours tas and for daily
file_360_hour = Dataset(three_hour)
file_360_hour_diskless = Dataset(three_hour_edit , mode='a')
tas_360 = file_360_hour.variables["tas"][:]
time_tas_360 = file_360_hour.variables["time"][:]

file_360_daily = Dataset(daily)
with open(daily, 'rb') as f:
    nc_bytes = f.read()
file_360_daily_diskless = Dataset(daily_edit , mode='a')
tos_360 = file_360_daily.variables["tos"][:]
time_tos_360 = file_360_daily.variables["time"][:]

# check if you have a leap year
year = 1850
if calendar.isleap(year):
    days = 366
else:
    days = 365

##############Transform-method1#################################
#!!! Pay attention if is less than daily frequency preserve the daily cycle
# Method 1: simply strech (this is the most straight forward way)
# here an example for one grid point only (in lat and lon) for simplicity
time_or = np.arange(1, 360 + 1, 1)
count = 360 / days
time_str_method1 = np.arange(count, 360 + count, count)

# for daily fields
tos_str_method1 = np.interp(time_str_method1, time_or, np.squeeze(tos_360[:, 0, 0]))
time_tos_365 = np.arange(1, days + 1, 1)
new_time_daily, new_tos_str_method1 = stretch_time(time_var=time_tos_360, data_var=tos_360, hourly_interval=24)

# for 3 hourly fields
n_m = int(24 / 3)  # number of measurments, every 3 hours
tas_str_method1 = np.empty(shape=(days * n_m))
for ih in range(0, n_m):
    tt = np.interp(time_str_method1, time_or, np.squeeze(tas_360[ih : time_tas_360.size : n_m, 0, 0]))
    tas_str_method1[ih : days * n_m : n_m] = tt

count = 1 / (n_m)
time_tas_365 = np.arange(count, days + count, count)
new_time_three_hour, new_tas_str_method1 = stretch_time(time_var=time_tas_360, data_var=tas_360, hourly_interval=3)

##################Transform-method2##########################
# Method 2: adds extra days
# some times is preferable depending on the application:
# e.g., if you want to go from no-leap to leap caledar it may be better to add an extra day
# in February rather than adjust/strech the whole year)
# as an example add 5 or 5 days here throughout the year

# for daily fields
# first and last 15 stay the same
time_str_method2 = np.arange(1, 15 + 1, 1)
count_n = int((days - 15 * 2) / (days - 360))
extra_days = days - 360
t_end = time_str_method2[-1]
for id in range(1, extra_days + 1):
    tt = np.arange(t_end + 1, count_n + t_end, 1)
    time_str_method2 = np.append(time_str_method2, tt)
    t_end = time_str_method2[-1]
    time_str_method2 = np.append(time_str_method2, (t_end + 0.5))

# add the last 15 days
tt = np.arange(time_str_method2[-1] + 0.5, 360 + 1, 1)
time_str_method2 = np.append(time_str_method2, tt)

tos_str_method2 = np.interp(time_str_method2, time_or, np.squeeze(tos_360[:, 0, 0]))
dataset = old_add_time(dataset= file_360_daily_diskless, hourly_interval=24)
xr_dataset = add_time(dataset= daily_dataset, hourly_interval=24)

# for 3 hourly fields
n_m = int(24 / 3)  # number of measurments, every 3 hours
tas_str_method2 = np.empty(shape=(days * n_m))
for ih in range(0, n_m):
    tt = np.interp(time_str_method2, time_or, np.squeeze(tas_360[ih : time_tas_360.size : n_m, 0, 0]))
    tas_str_method2[ih : days * n_m : n_m] = tt

#dataset = add_time(dataset= file_360_hour_diskless, hourly_interval=3)

#############Check things#########################
plt.plot(new_time_daily, new_tos_str_method1, "b.")
plt.plot(time_tos_365, tos_str_method1, "g.")
plt.plot(time_tos_360 * 365 / 360, np.squeeze(tos_360[:, 0, 0]), "r.")
plt.show()


plt.plot(new_time_three_hour, new_tas_str_method1, "b.")
plt.plot(time_tas_365, tas_str_method1, "g.")
plt.plot(time_tas_360 * 365 / 360, np.squeeze(tas_360[:, 0, 0]), "r.")
plt.show()


plt.plot(time_str_method2, tos_str_method2, "g.")
new_daily = [d[0][0] for d in xr_dataset['tos'][:]]
plt.plot(xr_dataset['time'][:],new_daily, "b.")
plt.plot(file_360_daily_diskless.variables['stretched_time'][:], file_360_daily_diskless.variables['tos_stretched'][:], "y.")
plt.plot(time_tos_360, np.squeeze(tos_360[:, 0, 0]), "r.")
plt.show()

plt.plot(time_tas_365, tas_str_method2, "g.")
plt.plot(file_360_hour_diskless.variables['stretched_time'][:], file_360_hour_diskless.variables['tas_stretched'][:], "b.")
plt.plot(time_tas_360 * 365 / 360, np.squeeze(tas_360[:, 0, 0]), "r.")
plt.show()

#plt.plot(time_str_method2, tas_str_method2[1 : 365 * 8 : 8], "g.")
#plt.plot(time_or, np.squeeze(tas_360[1 : 360 * 8 : 8, 0, 0]), "r.")
#plt.show()


# Compare old method results to new generalised method results.
print(np.array_equal(time_tos_365, new_time_daily)) 
print(np.array_equal(tas_str_method1, new_tas_str_method1))

print(np.array_equal(time_tas_365, new_time_three_hour))
print(np.array_equal(tos_str_method1, new_tos_str_method1))

print(np.array_equal(time_tos_365, file_360_daily_diskless.variables['stretched_time'][:]))
print(np.array_equal(time_tas_365, file_360_hour_diskless.variables['stretched_time'][:]))
print(np.array_equal(tas_str_method2, file_360_hour_diskless.variables['tas_stretched'][:]))
print(np.array_equal(tos_str_method2, file_360_daily_diskless.variables['tos_stretched'][:]))