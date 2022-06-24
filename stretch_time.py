import calendar
import traceback
import xarray as xr
import numpy as np
from pathlib import Path
import cftime
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt  # plotting

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
    
    points_in_data = int(time_var.size) #/ (24 / hourly_interval))
    measures_per_day = 24 / hourly_interval
    time_original = np.arange(1, points_in_data + 1, 1)
    extra_days = days - (points_in_data / measures_per_day)
    
    first_15 = np.arange(1,(15*measures_per_day) + 1)
    last_15 = np.arange(points_in_data - (15 * measures_per_day) + 1 , points_in_data + 1)
    extended_days = np.arange((15*measures_per_day) + 1, points_in_data - (15*measures_per_day) +1, (points_in_data - (30*measures_per_day))/(points_in_data - (30*measures_per_day) + (extra_days*measures_per_day)))
    extended_time = np.append(np.append(first_15, extended_days), last_15)
    
    
    # first_15 = np.arange(1,16)
    # last_15 = np.arange(360-14, 361)
    # 365 element array = np.append(np.append(first_15, np.arange(16, 346, 330/335)), last_15)
    # np.arange((15*2) + 1, points_in_data - (15*2) +1, (points_in_data - (30*2))/(points_in_data - (30*2) + (extra_days*2)))
    

    stretched_variables = []
    for var_name, data_var in dataset.variables.items():
        if time_var_name not in data_var.dims:
            continue
        try:

            interpolate1d = interp1d(time_original, y=data_var[:], axis=0) # Scipy interp retains all dims. (No np.squeeze)
            new_data = interpolate1d(extended_time)
               
            # Create new stretched variable. 
            dim_dict = {dim: dataset[dim][:] for dim in data_var.dims if dim != time_var_name}
            dim_dict['time'] = xr.cftime_range(
                start = daily_dataset['time'].data[0].isoformat(),
                end = daily_dataset['time'].data[-1].isoformat(),
                periods = extended_time.size,
                freq = None,
                calendar="all_leap"
            )
            # Create new data array for stretched variable.
            data_array = xr.DataArray(data=new_data, coords=dim_dict, name=var_name, dims=("time", "lat", "lon"))
            stretched_variables.append(data_array)
        except Exception as exc:
            print(f"{var_name} -- {traceback.format_exc()}")
        # Create new dataset from all stretched variables.
        new_dataset = xr.Dataset(data_vars={var.name: var for var in stretched_variables})
    return new_dataset, time_original, extended_time


old_ds = three_hour_dataset
new_ds, og_time, extended_time = add_time(old_ds, hourly_interval=3)
print(new_ds)
print(new_ds.time)
print(new_ds.tas.size)
new_data = [d[0][0] for d in new_ds['tas'][:]]
new_times = [d.isoformat() for d in new_ds['time'].data[:]]
old_times = [d.isoformat() for d in old_ds['time'].data[:]]
old_data = [d[0][0] for d in old_ds['tas'][:]]


plt.plot(extended_time, new_data, "g.")
plt.plot(og_time, old_data, 'r.')
plt.show()




