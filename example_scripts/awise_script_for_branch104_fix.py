#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:53:20 2020

@author: anwise
"""

import coast
import numpy as np
import xarray as xr

NEMO_PATH = 'example_files/nemo_data_'
DOMAIN_PATH = 'example_files/COAsT_example_NEMO_domain.nc'

ds_t = xr.open_dataset( NEMO_PATH + 'T_grid.nc' )
ds_u = xr.open_dataset( NEMO_PATH + 'U_grid.nc' )
ds_v = xr.open_dataset( NEMO_PATH + 'V_grid.nc' )

nemo_t = coast.NEMO( NEMO_PATH + 'T_grid.nc', DOMAIN_PATH, grid_ref='t-grid' )
nemo_u = coast.NEMO( NEMO_PATH + 'U_grid.nc', DOMAIN_PATH, grid_ref='u-grid' )
nemo_v = coast.NEMO( NEMO_PATH + 'V_grid.nc', DOMAIN_PATH, grid_ref='v-grid' )
nemo_f = coast.NEMO( DOMAIN_PATH, grid_ref='f-grid' )

nemo_t.dataset
nemo_f.dataset