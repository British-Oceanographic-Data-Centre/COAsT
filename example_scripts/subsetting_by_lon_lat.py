import numpy as np
import coast

lon = np.arange(10, 30)
checkDim = False
if checkDim:
    lat = np.arange(20, 41)  # Testing array size check
else:
    lat = lon  # = lon-10 would leave some selections empty in the following
print("The dataset is a set of vertices on a straight line in lon/lat space")
print("Here's what we are starting with - for longitude (= latitude)")
print(lon)
ind = coast.general_utils.subset_indices_lonlat_box(lon, lat, ss=15, ww=20, ee=25)
print("Here's the subset for [20,25] - note omission of 25")
print(lon[ind])
print(
    "Interchange east and west. This should go from 25E all the way around to 20E. But for some reason it misses out 10-14"
)
ind = coast.general_utils.subset_indices_lonlat_box(lon, lat, ss=15, ww=25, ee=20)
print(lon[ind])
print("Use negative longitude")
ind = coast.general_utils.subset_indices_lonlat_box(lon, lat, ss=15, ww=-340, ee=25)
print(lon[ind])
# print("Don't use same value for the two boundaries")
# ind = cst.general_utils.subset_indices_lonlat_box(lat,lon,ss=-15,ww=25,ee=-335)
# print(lon[ind])
