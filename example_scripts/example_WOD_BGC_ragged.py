#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:53:33 2022

@author: annkat
"""

#%reset
#====================== LOAD MODULES =========================
import sys # sys.exit('Error message) also module import
sys.path.append("/work/annkat/COAST_DEV/COAsT")
import coast
import glob # For getting file paths
import gsw
import matplotlib.pyplot as plt
import datetime
import numpy as np
import xarray as xr
import coast.general_utils as general_utils
import scipy as sp

#====================== UNIV PARAMS ===========================
path_examples = '/projectsa/COAsT/NEMO_example_data/SEAsia_R12/'   ## data local

#====================== load my data ===========================
fn_WOD_var = (path_examples + 'WOD_example_ragged_standard_level.nc')#'WOD_example_ragged_OBSdepth.nc')
fn_WOD_config = (path_examples + 'example_WOD_profiles.json')

WOD_profile_1D = coast.Profile(config=fn_WOD_config)
WOD_profile_1D.read_WOD(fn_WOD_var )

#===================== reshape TO 2D=====================
#choose which observed variables you want
VAR_USER_want=['Salinity','Temperature','Nitrate','Oxygen','DIC','Phosphate','Alkalinity']
WOD_profile = coast.Profile.reshape_2D(WOD_profile_1D, VAR_USER_want)

#===================== keep subset =====================
ind = WOD_profile.subset_indices_lonlat_box([90, 120], [-5, 5])[0]
WOD_profile = WOD_profile.isel(profile=ind)

#===================== SEAsia read BGC =====================
# note in this simple test nemo data are only for 3 months from 1990 so the 
# comparisons are not going to be correct but just as a demo

fn_SEAsia_domain = (path_examples + 'coast_example_domain_SEAsia.nc')
fn_SEAsia_config_BGC = (path_examples + 'example_nemo_BGC.json')
fn_SEAsia_var = (path_examples + 'coast_example_SEAsia_BGC_1990.nc')

SEAsia_BGC = coast.Gridded(fn_data = fn_SEAsia_var, fn_domain = fn_SEAsia_domain, 
                       config = fn_SEAsia_config_BGC, multiple=True)

#My domain file does not have mask so this is just a trick 
SEAsia_BGC.dataset["landmask"] = SEAsia_BGC.dataset.bottom_level == 0
SEAsia_BGC.dataset = SEAsia_BGC.dataset.rename({"depth_0": "depth"})
model_profiles = WOD_profile.obs_operator(SEAsia_BGC)

#remove any points that are farmodel
too_far = 5
keep_indices = model_profiles.dataset.interp_dist <= too_far
model_profiles = model_profiles.isel(profile=keep_indices)
WOD_profile = WOD_profile.isel(profile=keep_indices)

#transform observed DIC from mmol/l to mmol C/ m^3 that the model has
fig = plt.figure()
plt.plot(1000*WOD_profile.dataset.DIC[8,:],WOD_profile.dataset.depth[8,:], linestyle="", marker="o")
plt.plot(model_profiles.dataset.DIC[8,:], model_profiles.dataset.depth[:,8],linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title('DIC vs depth')
plt.show()

fig = plt.figure()
plt.plot(WOD_profile.dataset.Oxygen[8,:],WOD_profile.dataset.depth[8,:], linestyle="", marker="o")
plt.plot(model_profiles.dataset.oxygen[8,:], model_profiles.dataset.depth[:,8],linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title('Oxygen vs depth')
plt.show()

fig = plt.figure()
plt.plot(WOD_profile.dataset.Phosphate[8,:],WOD_profile.dataset.depth[8,:], linestyle="", marker="o")
plt.plot(model_profiles.dataset.phosphate[8,:], model_profiles.dataset.depth[:,8],linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title('Phosphate vs depth')
plt.show()


#interpolate SEAsia to profile depths
reference_depths = WOD_profile.dataset.depth[20,:].values
model_profiles.dataset = model_profiles.dataset[["DIC"]]/1000
model_profiles_interp = model_profiles.interpolate_vertical(reference_depths, interp_method="linear")

#!!NOTE INTERPOLATE does not work with WOD_profiles (maybe due to all nans in some points)
#WOD_profile.dataset = WOD_profile.dataset[["DIC", "depth"]]
#WOD_interp = WOD_profile.interpolate_vertical(reference_depths, interp_method="linear")

#calculate differences
differences= WOD_profile.difference(model_profiles_interp)
differences.dataset.load()

