
import xarray as xr
import cftime
from pathlib import Path

data_dir = Path("E:\\COAsT data\\")
daily = data_dir / "tos_Oday_UKESM1-0-LL_historical_r1i1p1f2_gn_1850_subset.nc"

daily_dataset = xr.load_dataset(daily)
new_dataset = daily_dataset.convert_calendar("365_day", "time", align_on = "date")

print(daily_dataset.time[:][0])



# Newer
date_range = xr.cftime_range(
    start = daily_dataset['time'].data[0].isoformat(),
    end = daily_dataset['time'].data[-1].isoformat(),
    calendar="all_leap"
    )

old_range = [d.strftime() for d in daily_dataset['time'].data[:]]
new_range = [d.strftime() for d in date_range]

#newer_data = daily_dataset.interp_calendar(date_range, "time")
print(len(old_range))
print(len(new_range))

print(old_range[:10])
print(new_range[:10])




# 360 to 365
# numberOfYears = a360
# start = a360[0]
# length = a360.length()
# a365 = data.range(...)