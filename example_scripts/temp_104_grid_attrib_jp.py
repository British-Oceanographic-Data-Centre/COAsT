# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
In the absence of a unit test
"""

import coast
import numpy as np

dir = 'example_files/'
fn_dom = dir + 'COAsT_example_NEMO_domain.nc'
fn_dat = dir + 'COAsT_example_NEMO_data.nc'

#%%
sci = coast.NEMO(fn_dat, fn_dom, grid_ref='t-grid')


#%%
# Build coords for random variables

np.random.seed(123)

temperature_3d = 15 + 10 * np.random.randn(2,5,4,3)    # 3-dimensional
lat = np.random.uniform(low=50, high=60, size=(4,3))
lon = np.random.uniform(low=-5, high=10, size=(4,3))
depth_t =  np.random.uniform(low=20, high=40, size=(5,4,3))

# round to two digits after decimal point
temperature_3d = np.around(temperature_3d, decimals=2)
lat , lon = np.around([lat, lon], decimals=2)


da = xr.DataArray(data=temperature_3d,
                  coords={"lat": (["y_dim","x_dim"], lat),
                          "lon": (["y_dim","x_dim"], lon),
                          "dep": (["z_dim","y_dim","x_dim"], depth_t),
                          "t_dim": [88,89]},
                  dims=["t_dim","z_dim","y_dim","x_dim"])

#print(da)

#%%

da.sel(t_dim=slice(70, 88.8)) # slice by value
da.sel(z_dim=slice(2, 3)) # slice by level

da[0,:,:,:].where(da[0,:,:,:].dep < 30)  # find by depth values
