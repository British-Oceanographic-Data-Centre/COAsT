#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:10:36 2022

@author: jrule

This example reads in CMEMS model data (parent model) and SRIL34 model data (child model).
The CMEMS data is subsetted within the coordinates of the SRIL34 model. As example, the salinity data
is interpolated horizontally and vertically. Because the SRIL34 model uses hybrid vertical coordinates,
the vertical interpolation is done point by point (this can be slow). 
Example figures of the original mode, interpolated model and difference between model are plotted.
"""
#################################################
# 1. Import libraries
#################################################
import coast

# from coast import profile
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import numpy as np
from datetime import datetime
import pandas as pd

#################################################
# choose date to compare
#################################################
tt = datetime(2020, 1, 1)
tp = tt.strftime("%Y_%m_%d")
# print("time:", tp)

#################################################
# Loading  data
#################################################
# first model
parent_dir = "/projectsa/COAsT/CMEMS/"
parent_filet = "coast_example_cmems_" + tp + "_download_T.nc"
parent_fileuv = "coast_example_cmems_" + tp + "_download_UV.nc"
parent_config_t = "../config/example_cmems_grid_t.json"  # make new config file
parent_config_uv = "../config/example_cmems_grid_uv.json"

# second model
child_dir = "/projectsa/COAsT/SRIL34/"
child_filet = "coast_example_sril34_1d_T.nc"
child_fileu = "coast_example_sril34_1d_U.nc"
child_filev = "coast_example_sril34_1d_V.nc"
child_dom_nam = "/projectsa/COAsT/SRIL34/domain_cfg.nc"
child_config_t = "../config/example_nemo_grid_t.json"
child_config_u = "../config/example_nemo_grid_u.json"
child_config_v = "../config/example_nemo_grid_v.json"

########################################################
# grid data with coast
########################################################
parent_sci_t = coast.Gridded(parent_dir + parent_filet, config=parent_config_t)
parent_sci_uv = coast.Gridded(parent_dir + parent_fileuv, config=parent_config_uv)

child_sci_t = coast.Gridded(child_dir + child_filet, child_dom_nam, config=child_config_t)
child_sci_u = coast.Gridded(child_dir + child_fileu, child_dom_nam, config=child_config_u)
child_sci_v = coast.Gridded(child_dir + child_filev, child_dom_nam, config=child_config_v)

########################################################
# define subset to match the model domains
########################################################
# subset of coordinates based on the smaller (child) model
minx = child_sci_t.dataset.longitude.min()
maxx = child_sci_t.dataset.longitude.max()
miny = child_sci_t.dataset.latitude.min()
maxy = child_sci_t.dataset.latitude.max()

# subset the bigger (parent) model
ind_sci = parent_sci_t.subset_indices([miny, minx], [maxy, maxx])
sci_t_box = parent_sci_t.isel(latitude=ind_sci[0], longitude=ind_sci[1])

###########################################################
#  2d horizontal interp only on surface level
###########################################################
# get parent data to interpolate
sal = sci_t_box.dataset.salinity.isel(t_dim=0).isel(z_dim=0)  # t alway 0 as you only have 1 CMEMS
z = np.array(sal)
xp, yp = np.meshgrid(sci_t_box.dataset.longitude, sci_t_box.dataset.latitude)

# query point to interpolate on (child model grid)
xx = child_sci_t.dataset.longitude
yy = child_sci_t.dataset.latitude

# interpolate
zz = griddata((xp.ravel(), yp.ravel()), z.ravel(), (xx, yy), method="linear")  # 2d horizontal, one level

###########################################################
# 2d horizontal interp on all z levels (loop)
###########################################################

# loop through levels, inperpolate and append to list
pzl = len(parent_sci_t.dataset.z_dim)  # parent z levels to set loop
list0 = []
for n in range(pzl):
    sal1 = sci_t_box.dataset.salinity.isel(t_dim=0).isel(z_dim=n)
    z1 = np.array(sal1)
    zz1 = griddata((xp.ravel(), yp.ravel()), z1.ravel(), (xx, yy), method="linear")  # 2d horizontal, one level
    list0.append(zz1.copy())
# stack list to a 3d np.array
horzinterp = np.dstack(list0)

###########################################################
# vertical interpolation at all points
###########################################################
# with hybrid coordinates depth chages at each horizontal grid point
depth = child_sci_t.dataset.depth_0  # depth from hybrid coordinates

list1 = []
for yi in range(len(xx)):
    for xi in range(len(xx[0])):
        # print(xi)
        # print(yi)
        childz = child_sci_t.dataset.depth_0.isel(y_dim=yi, x_dim=xi)
        parentz = parent_sci_t.dataset.z_dim
        pov = horzinterp[yi, xi, :]  # parent model old value (after horizontal interp)
        f = interp1d(parentz, pov, bounds_error=False, fill_value="extrapolate")
        pnv = f(childz)  # parent model new value after vertical interp
        list1.append(pnv.copy())
varinterp = np.reshape(list1, (len(xx), len(xx[0]), len(depth)))

########################################################
# get the selected date from the child model
########################################################
date = pd.DatetimeIndex.strftime(pd.DatetimeIndex(child_sci_t.dataset.time), "%Y_%m_%d")
ti = np.squeeze(np.where(date == tp))

########################################################
# plots to compare at specific times and depth
########################################################
# mask = np.ma.masked_where(varinterp == 0, varinterp)
# choose level to plot
lp = 0  # z level for parent model
li = 0  # z level for child model or interpolated data

# parent subset - salinity
fig = plt.figure()
plt.pcolormesh(
    sci_t_box.dataset.longitude, sci_t_box.dataset.latitude, sci_t_box.dataset.salinity.isel(t_dim=0).isel(z_dim=lp)
)
plt.clim(30, 35)
plt.colorbar()
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Original CMEMS, salinity (psu), z level = " + str(lp))

# child salinity
fig = plt.figure()
plt.pcolormesh(
    child_sci_t.dataset.longitude,
    child_sci_t.dataset.latitude,
    child_sci_t.dataset.salinity.isel(t_dim=ti).isel(z_dim=li),
)
plt.colorbar()
plt.clim(30, 35)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Original SRIL34, salinity (psu), z level = " + str(li))

# interpolated data
fig = plt.figure()
plt.pcolormesh(xx, yy, varinterp[:, :, li])
plt.colorbar()
plt.clim(30, 35)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Interpolated CMEMS, salinity (psu), z level = " + str(li))

# differences plot
fig = plt.figure()
plt.pcolormesh(
    child_sci_t.dataset.longitude,
    child_sci_t.dataset.latitude,
    child_sci_t.dataset.salinity.isel(t_dim=ti).isel(z_dim=li) - varinterp[:, :, li],
)
plt.clim(-1, 1)
plt.colorbar()
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Differences, salinity (psu), z level = " + str(li))
