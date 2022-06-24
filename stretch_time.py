import calendar
import traceback
import xarray as xr
import numpy as np
from pathlib import Path
import cftime
from scipy.interpolate import interp1d
from datetime import datetime

data_dir = Path("E:\\COAsT data\\")
three_hour = data_dir / "tas_3hr_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"
daily = data_dir / "tos_Oday_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"


daily_dataset = xr.load_dataset(daily)
three_hour_dataset = xr.load_dataset(three_hour)

# TODO: Convert stretch time to xrray usage. Also to retain dims.
def add_time(dataset: xr.Dataset, time_var_name: str = "time", year: int = 1850, hourly_interval: int = 24):
    """Add days to allow add leap year support."""
    
    time_var = dataset[time_var_name][:]
    
    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    
    days_in_data = int(time_var.size / (24 / hourly_interval))
    time_original = np.arange(1, days_in_data + 1, 1)
    
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
    
    # 360 days - 30 = 330 days
    # 24 hours / hourly interval = number of measures a day (e.g. 3hrly = 24 hours / 3 = 8 measurements per day)
    # add 365 - 360 = 5 days
    # 5 / (330 * number of measures per day) this is a fraction of a day that should be added to each day?
    #### 330 / 5 = 66 so every 66 days add an element (day)
    # [1 - 360] length = 360
    # [1, 2, 2+val ...., 360] 
    #  1 to 330 step value = 5 / (330 * number of measures per day)


    # [1 - 365] step = 360/365
    # first_15 = np.arange(1,16)
    # last_15 = np.arange(360-14, 361)
    # 365 element array = np.append(np.append(first_15, np.arange(16, 346, 330/335)), last_15)
    

    stretched_variables = []
    for var_name, data_var in dataset.variables.items():
        if time_var_name not in data_var.dims:
            continue
        try:

            interpolate1d = interp1d(time_original, y=data_var[:], axis=0) # Scipy interp retains all dims. (No np.squeeze)
            new_data = interpolate1d(time_str_method2)
               
            # Create new stretched variable. 
            dim_dict = {dim: dataset[dim][:] for dim in data_var.dims if dim != time_var_name}
            dim_dict['time'] = xr.cftime_range(
                start = daily_dataset['time'].data[0].isoformat(),
                end = daily_dataset['time'].data[-1].isoformat(),
                calendar="all_leap"
            )
            # Create new data array for stretched variable.
            data_array = xr.DataArray(data=new_data, coords=dim_dict, name=var_name, dims=("time", "i", "j"))
            stretched_variables.append(data_array)
        except Exception as exc:
            print(f"{var_name} -- {traceback.format_exc()}")
        # Create new dataset from all stretched variables.
        new_dataset = xr.Dataset(data_vars={var.name: var for var in stretched_variables})
    return new_dataset


new_ds = add_time(daily_dataset)
#new_ds = add_time(three_hour_dataset, hourly_interval=3)
print(new_ds)
print(new_ds.time)
print(new_ds.tos.size)



