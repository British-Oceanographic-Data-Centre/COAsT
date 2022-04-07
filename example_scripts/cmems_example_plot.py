#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:35:37 2022
example to load and plot CMEMS data. Note that the example files 
coast_example_cmems_2020_01_01_*.nc are cropped to reduce the file size (3<lat<12 and 75<lon<85 aproximatly).
@author: jrule
"""

#################################################
# 1. Import libraries
#################################################
import coast
import matplotlib.pyplot as plt

#################################################
# Loading  data
#################################################
dir_nam = "/projectsa/COAsT/CMEMS/"
filet = "coast_example_cmems_2020_01_01_download_T.nc"
fileuv = "coast_example_cmems_2020_01_01_download_UV.nc"
config_t = "../config/example_cmems_grid_t.json"#make new config file
config_uv = "../config/example_cmems_grid_uv.json"

sci_t = coast.Gridded(dir_nam + filet, config=config_t)
sci_uv = coast.Gridded(dir_nam + fileuv, config=config_uv)


#################################################
#calculate currents
#################################################
current = abs(sci_uv.dataset.u_velocity)*abs(sci_uv.dataset.v_velocity)

#################################################
#plot
#################################################
#choose time step and depth levels to plot
li=0 #z level

# full domain salinity
fig = plt.figure()
plt.pcolormesh(sci_t.dataset.longitude, sci_t.dataset.latitude, sci_t.dataset.salinity.isel(t_dim=0).isel(z_dim=li))
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("CMEMS, surface salinity (psu)")

#full domain currents
fig = plt.figure()
plt.pcolormesh(sci_t.dataset.longitude, sci_t.dataset.latitude, current.isel(t_dim=0).isel(z_dim=li))
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("CMEMS, surface currents (m/s)")
