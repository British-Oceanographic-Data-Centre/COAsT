import traceback
from typing import List
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt  # plotting
import pandas as pd

data_dir = Path("E:\\COAsT data\\")
three_hour = data_dir / "tas_3hr_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"
daily = data_dir / "tos_Oday_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"


daily_dataset = xr.load_dataset(daily)
three_hour_dataset = xr.load_dataset(three_hour)

def get_start_year(time_data: np.ndarray) -> int:
    return time_data[0].timetuple()[0]

def get_end_year(time_data: np.ndarray) -> int:
    last_day = time_data[-1].timetuple()
    if last_day[1:6] == (1,1,0,0,0):
        return last_day[0] - 1
    return time_data[-1].timetuple()[0]

def get_date_range(time_data: np.ndarray, hourly_interval) -> List:
    return pd.date_range(
        f"{get_start_year(time_data)}-01-01 00:00:00",
        f"{get_end_year(time_data)}-12-31 23:59:59",
        freq= f"{hourly_interval}H"
        ).tolist()

def extend_number_of_days(points_in_data: int, measures_per_day: int, extra_days: int) -> np.ndarray:
    # Expand 330 days to 335/336 days (365 days - 30 ignored days.) 
    
    first_15 = np.arange(1,(15*measures_per_day) + 1)   # Ignore first 15 days.
    last_15 = np.arange(points_in_data - (15 * measures_per_day) + 1 , points_in_data + 1)  # Ignore last 15 days.
    extended_days = np.arange(
        (15*measures_per_day) + 1,
        points_in_data - (15*measures_per_day) +1,
        (points_in_data - (30*measures_per_day))/(points_in_data - (30*measures_per_day) + (extra_days*measures_per_day))
        )
    extended_time = np.append(np.append(first_15, extended_days), last_15)
    
    return extended_time


def add_time(dataset: xr.Dataset, time_var_name: str = "time", hourly_interval: int = 24):
    """Add days to allow add leap year support."""
    
    time_data = dataset[time_var_name].data
    date_range = get_date_range(time_data, hourly_interval)
    measures_per_day = 24 / hourly_interval
    days = len(date_range) / measures_per_day
     
    points_in_data = int(len(time_data))
    
    time_original = np.arange(1, points_in_data + 1, 1)
    extra_days = days - (points_in_data / measures_per_day)
    
    extended_time = extend_number_of_days(
        points_in_data= points_in_data,
        measures_per_day= measures_per_day,
        extra_days= extra_days
    )
      
    stretched_variables = []
    for var_name, data_var in dataset.variables.items():
        if time_var_name not in data_var.dims:
            continue
        try:
            
            # Interpolate data.
            interpolate1d = interp1d(time_original, y=data_var[:], axis=0) # Scipy interp retains all dims. (No np.squeeze)
            new_data = interpolate1d(extended_time)
               
            # Create new stretched variable. 
            dim_dict = {dim: dataset[dim][:] for dim in data_var.dims if dim != time_var_name}
            dim_dict[time_var_name] = xr.cftime_range(
                start = dataset[time_var_name].data[0].isoformat(),
                end = dataset[time_var_name].data[-1].isoformat(),
                periods = extended_time.size,
                freq = None,
                calendar="all_leap"
            )
            # Create new data array for stretched variable.
            data_array = xr.DataArray(data=new_data, coords=dim_dict, name=var_name, dims= data_var.dims)
            stretched_variables.append(data_array)
        except Exception:
            print(f"{var_name} -- {traceback.format_exc()}")
        # Create new dataset from all stretched variables.
        new_dataset = xr.Dataset(data_vars={var.name: var for var in stretched_variables})
    return new_dataset, time_original, extended_time


old_ds = three_hour_dataset
new_ds, og_time, extended_time = add_time(old_ds, hourly_interval=3)

print(new_ds.time)

new_data = [d[0][0] for d in new_ds['tas'].data[:]]
new_times = [d.isoformat() for d in new_ds['time'].data[:]]
old_times = [d.isoformat() for d in old_ds['time'].data[:]]
old_data = [d[0][0] for d in old_ds['tas'].data[:]]

print(len(old_data))
print(len(new_data))


plt.plot(og_time, old_data, 'r.')
plt.plot(extended_time, new_data, "g.")

plt.show()




