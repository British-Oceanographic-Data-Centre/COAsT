# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
In the absence of a unit test
"""

import coast


dir = 'example_files/'
fn_dom = dir + 'COAsT_example_NEMO_domain.nc'
fn_dat = dir + 'COAsT_example_NEMO_data.nc'

#%%
sci = coast.NEMO(fn_dat, fn_dom, grid_ref='t-grid')

#%%
nemo_dom = coast.DOMAIN()
nemo_var = coast.NEMO()

nemo_dom.load(fn_dom)
nemo_var.load(fn_dat)

nemo_var.set_command_variables()
nemo_dom.set_command_variables()
