#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:53:33 2022

@author: annkat
"""

#%reset
# ====================== LOAD MODULES =========================
import coast
import glob  # For getting file paths
import gsw
import matplotlib.pyplot as plt
import datetime
import numpy as np
import xarray as xr
import coast.general_utils as general_utils
import scipy as sp

# ====================== UNIV PARAMS ===========================
path_examples = "./example_files/"  
## data local in livljobs : /projectsa/COAsT/NEMO_example_data/SEAsia_R12/
path_config = "./config/"

# ====================== load my data ===========================
fn_wod_var = path_examples + "WOD_example_ragged_standard_level.nc"  #'wod_example_ragged_OBSdepth.nc')
fn_wod_config = path_config + "example_wod_profiles.json"

wod_profile_1d = coast.Profile(config=fn_wod_config)
wod_profile_1d.read_wod(fn_wod_var)

# ===================== reshape TO 2D=====================
# choose which observed variables you want
var_user_want = ["salinity", "temperature", "nitrate", "oxygen", "dic", "phosphate", "alkalinity"]
wod_profile = coast.Profile.reshape_2d(wod_profile_1d, var_user_want)

# ===================== keep subset =====================
ind = wod_profile.subset_indices_lonlat_box([90, 120], [-5, 5])[0]
wod_profile = wod_profile.isel(profile=ind)

# ===================== SEAsia read BGC =====================
# note in this simple test nemo data are only for 3 months from 1990 so the
# comparisons are not going to be correct but just as a demo

fn_seasia_domain = path_examples + "coast_example_domain_SEAsia.nc"
fn_seasia_config_bgc = path_config + "example_nemo_bgc.json"
fn_seasia_var = path_examples + "coast_example_SEAsia_BGC_1990.nc"

seasia_bgc = coast.Gridded(
    fn_data=fn_seasia_var, fn_domain=fn_seasia_domain, config=fn_seasia_config_bgc, multiple=True
)

# My domain file does not have mask so this is just a trick
seasia_bgc.dataset["landmask"] = seasia_bgc.dataset.bottom_level == 0
seasia_bgc.dataset = seasia_bgc.dataset.rename({"depth_0": "depth"})
model_profiles = wod_profile.obs_operator(seasia_bgc)

# remove any points that are farmodel
too_far = 5
keep_indices = model_profiles.dataset.interp_dist <= too_far
model_profiles = model_profiles.isel(profile=keep_indices)
wod_profile = wod_profile.isel(profile=keep_indices)

# transform observed DIC from mmol/l to mmol C/ m^3 that the model has
fig = plt.figure()
plt.plot(1000 * wod_profile.dataset.dic[8, :], wod_profile.dataset.depth[8, :], linestyle="", marker="o")
plt.plot(model_profiles.dataset.dic[8, :], model_profiles.dataset.depth[:, 8], linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title("DIC vs depth")
plt.show()

fig = plt.figure()
plt.plot(wod_profile.dataset.oxygen[8, :], wod_profile.dataset.depth[8, :], linestyle="", marker="o")
plt.plot(model_profiles.dataset.oxygen[8, :], model_profiles.dataset.depth[:, 8], linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title("Oxygen vs depth")
plt.show()

fig = plt.figure()
plt.plot(wod_profile.dataset.phosphate[8, :], wod_profile.dataset.depth[8, :], linestyle="", marker="o")
plt.plot(model_profiles.dataset.phosphate[8, :], model_profiles.dataset.depth[:, 8], linestyle="", marker="o")
plt.ylim([2500, 0])
plt.title("Phosphate vs depth")
plt.show()


# interpolate seasia to profile depths
reference_depths = wod_profile.dataset.depth[20, :].values
model_profiles.dataset = model_profiles.dataset[["dic"]] / 1000
model_profiles_interp = model_profiles.interpolate_vertical(reference_depths, interp_method="linear")

#!!NOTE INTERPOLATE does not work with wod_profiles (maybe due to all nans in some points)
# wod_profile.dataset = wod_profile.dataset[["DIC", "depth"]]
# wod_interp = wod_profile.interpolate_vertical(reference_depths, interp_method="linear")

# calculate differences
differences = wod_profile.difference(model_profiles_interp)
differences.dataset.load()
