#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:53:20 2020

@author: anwise
"""

import coast
import os
import numpy as np
import xarray as xr

dn_files = "./example_files/"
dn_fig = 'unit_testing/figures/'
fn_nemo_grid_t_dat = 'nemo_data_T_grid.nc'
fn_nemo_grid_u_dat = 'nemo_data_U_grid.nc'
fn_nemo_grid_v_dat = 'nemo_data_V_grid.nc'
fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
fn_altimetry = 'COAsT_example_altimetry_data.nc'

if not os.path.isdir(dn_files):
    print("please go download the examples file from https://dev.linkedsystems.uk/erddap/files/COAsT_example_files/")
    dn_files = input("what is the path to the example files:\n")
    if not os.path.isdir(dn_files):
        print(f"location f{dn_files} cannot be found")

nemo_t = coast.NEMO( fn_data=dn_files+fn_nemo_grid_t_dat, 
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='t-grid' )
nemo_u = coast.NEMO( fn_data=dn_files+fn_nemo_grid_u_dat, 
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='u-grid' )
nemo_v = coast.NEMO( fn_data=dn_files+fn_nemo_grid_v_dat, 
                    fn_domain=dn_files+fn_nemo_dom, grid_ref='v-grid' )
nemo_f = coast.NEMO( fn_domain=dn_files+fn_nemo_dom, grid_ref='f-grid' )

tran = coast.Transect( (54,-15), (56,-12), nemo_f, nemo_t, nemo_u, nemo_v )