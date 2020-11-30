#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:49:58 2020

@author: awise
"""
import xarray as xr
import numpy as np
from scipy import interpolate

# for it in self.data.t_dim:
#     for ir in self.data.r_dim:
#         if not np.all(np.isnan(salinity_s[it,:,ir].data)):  
#             # Need to remove the levels below the (envelope) bathymetry which are NaN
#             salinity_s_r = salinity_s[it,:,ir].compressed()
#             temperature_s_r = temperature_s[it,:,ir].compressed()
#             s_levels_r = s_levels[:len(salinity_s_r),ir]
            
#             sal_func = interpolate.interp1d( s_levels_r, salinity_s_r, 
#                          kind='linear', fill_value="extrapolate")
#             temp_func = interpolate.interp1d( s_levels_r, temperature_s_r, 
#                          kind='linear', fill_value="extrapolate")
            
#             if extrapolate is True:
#                 salinity_z[it,:,ir] = sal_func(z_levels)
#                 temperature_z[it,:,ir] = temp_func(z_levels)                        
#             else:
#                 # set levels below the bathymetry to nan
#                 salinity_z[it,:,ir] = np.where( z_levels <= self.data.bathymetry.values[ir], 
#                         sal_func(z_levels), np.nan )
#                 temperature_z[it,:,ir] = np.where( z_levels <= self.data.bathymetry.values[ir], 
#                         temp_func(z_levels), np.nan ) 
    
                
PATH_SRC = '/Users/awise/Documents/Work/JMMP/VERT_COORD/sz_L51_r10_s21_1m_20050101_20051231_grid_T.nc'
PATH_DST = '/Users/awise/Documents/Work/JMMP/VERT_COORD/sf_L51_r24_1m_20050101_20051231_grid_T.nc' 
t_ind=0   
ds_src = coast.NEMO(PATH_SRC, grid_ref='t-grid').dataset.isel(t_dim = t_ind)  
ds_dst = coast.NEMO(PATH_DST, grid_ref='t-grid').dataset.isel(t_dim = t_ind) 
# Assuming we just do this for 1 time           
#ds_src = xr.open_dataset(PATH_SRC).isel(time = t_ind)  
#ds_dst = xr.open_dataset(PATH_DST).isel(time = t_ind)                 
thickness = ds_dst.e3t[0,:,:]
e3t_s_cum = ds_src.e3t.cumsum(dim=z)

# for surface
dz = thickness - ds_src.e3t[0,:,:]
if dz >= 0
var_dst[0,:,:] = dz * ds_src['var'][0,:,:]

for z_ind in ds_src.z_dim[1:]:
    for x_ind in ds_src.x_dim:
        for y_ind in ds_src.y_dim:        
            e3t_s = ds_src.e3t[z_ind,y_ind,x_ind]
            dz = 
            var_s = ds_src['var'][z_ind,y_ind,x_ind]
            var_d = e3t_s * var_s
    
z_0 = 0
z_1 = ds_dst.e3t[0,:,:]

e3t_s = ds_src.e3t
e3t_s_c = ds_src.e3t.cumsum(dim=z_dim)
if z_0 == 0:
    for z_ind in e3t_s_c.z_dim.data:
        if e3t_s_c[z_ind,y_ind,x_ind] <= z_1:
            dz = e3t_s_c[z_ind,y_ind,x_ind]
            var_d = dz * var_s[z_ind,y_ind,x_ind]
        else:
            dz = z_1
            var_d = dz * var_s[z_ind,y_ind,x_ind]
            
for z_ind in e3t_s_c.z_dim.data:
    if e3t_s_c[z_ind,y_ind,x_ind] <= z_0 and e3t_s_c[z_ind+1,y_ind,x_ind] > z_0:
        
e3t_dst_cum = ds_dst.e3t.cumsum(dim=z_dim)
dz_dst = e3t_dst_cum[1:,:,:] - e3t_dst_cum[:-1,:,:]


depth_w_dst = xr.zeros_like(ds_dst.e3t)
depth_w_dst[dict(z_dim=slice(1,None))] = ds_dst.e3t.cumsum(dim='z_dim')[dict(z_dim=slice(0,-1))]
depth_w_src = xr.zeros_like(ds_src.e3t)
depth_w_src[dict(z_dim=slice(1,None))] = ds_src.e3t.cumsum(dim='z_dim')[dict(z_dim=slice(0,-1))]

level = 0 #loop
z_t = depth_w_dst.isel(z_dim=level)
z_b = depth_w_dst.isel(z_dim=level+1)
if z_t == 0:
    var_d = 0
    for z_ind in ds_src.z_dim.values[1:]:
        if depth_w_src[z_ind,y_ind,x_ind] <= z_b:
            dz = ds_src.e3t[z_ind-1,y_ind,x_ind]
            var_d = var_d + dz * var_s[z_ind-1,y_ind,x_ind]
        else:
            dz = z_b - depth_w_src[z_ind-1,y_ind,x_ind]
            var_d = var_d + dz * var_s[z_ind,y_ind,x_ind]
            break
return var_d
else:
    var_d = 0
    for z_ind in ds_src.z_dim.values[1:]:
        if depth_w_src[z_ind,y_ind,x_ind] > z_t:
            if depth_w_src[z_ind,y_ind,x_ind] <= z_b:
                if depth_w_src[z_ind-1,y_ind,x_ind] <= z_t
                    dz = depth_w_src[z_ind,y_ind,x_ind] - z_t
                else:
                    dz = ds_src.e3t[z_ind-1,y_ind,x_ind]
                var_d = var_d + dz * var_s[z_ind-1,y_ind,x_ind]
            else: 
                if depth_w_src[z_ind-1,y_ind,x_ind] <= z_t:
                    dz = z_b - z_t
                else:
                    dz = z_b - depth_w_src[z_ind-1,y_ind,x_ind] 
                var_d = var_d + dz * var_s[z_ind-1,y_ind,x_ind]
                break
            
        
            
                
                
                
                
            



