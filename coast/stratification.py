#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:07:53 2020

@author: awise
"""

import xarray as xr
import numpy as np
import gsw
from seawater import eos80 as sw

def approx_depth_t(ds : xr.Dataset):
    depth_w = xr.zeros_like(ds.e3t)    
    depth_w[dict(z_dim=slice(1,None))] = ds.e3t.cumsum(dim='z_dim').isel(z_dim=slice(0,-1))
    depth_w = depth_w.assign_coords({'k': ('z_dim', ds.z_dim.data)})
    e3w = depth_w.differentiate('k',edge_order=2)
    depth_t = xr.full_like(depth_w, np.nan)
    depth_t[dict(z_dim=0)] = e3w.isel(z_dim=0) * 0.5
    depth_t[dict(z_dim=slice(1,None))]  = e3w.cumsum(dim='z_dim').isel(z_dim=slice(1,None)) \
                                        + depth_t[dict(z_dim=0)]
    depth_t = depth_t.drop('k')
    return depth_t
    
def pot_energy_anom(nemo_t: xr.Dataset, teos10=True):
    g = 9.81
    ds = nemo_t.dataset
    # get the approximate z coordinate (= -depth) for t-points 
    z_t = -approx_depth_t(ds)
    if teos10==True:
        # Approx pressure from depth
        pressure = gsw.p_from_z( z_t, ds.latitude )
        # In situ density using TEOS-10, assumes absolute salinity and conservative temperature
        density = gsw.rho( ds.salinity, ds.temperature, pressure )
    else:
        # Approx pressure from depth
        pressure = sw.pres( -z_t, ds.latitude )
        # In situ density using EOS80, assumes practical salinity and temperature
        density = sw.dens( ds.salinity, ds.temperature, pressure )
    # get the water column thickness
    thickness = ds.e3t.sum(dim='z_dim',skipna=True).data   
    # depth average density
    density_depthavg = (density*ds.e3t).sum(dim='z_dim') / thickness
        
    PEA = (g/thickness) * (( ( density_depthavg.data[:,np.newaxis,:,:] - density ) * z_t ) * ds.e3t ).sum(dim='z_dim')
    PEA.name='Potential Energy Anomaly'

    