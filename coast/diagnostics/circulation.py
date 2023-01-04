#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:47:59 2022

@author: jholt
"""
from ..data.gridded import Gridded
import numpy as np
import xarray as xr
import matplotlib.pylab as plt


class CurrentsonT(Gridded):
    """
    Methods for currents co-located on T-points

    use like coast.Gridded () with a domain_cfg file name to create gridded object

    """

    def __init(self, fn_domain=None, config=None, **kwargs):
        gridded = Gridded(fn_domain=fn_domain, config=config, **kwargs)
        self.dataset = gridded.dataset

    def currents_on_T(self, gridded_u, gridded_v):

        """
        Adds co-located velocity components and speed to CurrentsonT object
        Parameters
        ----------
        gridded_u : TYPE
            DESCRIPTION.
        gridded_v : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # U velocity on T points
        UT = np.zeros(gridded_u.dataset.u_velocity.shape)
        UT[:, :, :, 1:] = 0.5 * (
            gridded_u.dataset.u_velocity.values[:, :, :, 1:] + gridded_u.dataset.u_velocity.values[:, :, :, :-1]
        )
        # V velocity on T points
        VT = np.zeros(gridded_v.dataset.v_velocity.shape)
        VT[:, :, 1:, :] = 0.5 * (
            gridded_v.dataset.v_velocity.values[:, :, 1:, :] + gridded_v.dataset.v_velocity.values[:, :, :-1, :]
        )

        speed = np.sqrt(UT * UT + VT * VT)

        dims = gridded_u.dataset.u_velocity.dims
        self.dataset["ut_velocity"] = xr.DataArray(UT, dims=dims)
        self.dataset["vt_velocity"] = xr.DataArray(VT, dims=dims)
        self.dataset["speed_t"] = xr.DataArray(speed, dims=dims)

    def plot_surface_circulation(self, name, Vmax=0.16, Np=3, headwidth=4, scale=50, **kwargs):
        #%%
        from matplotlib import cm
        from matplotlib.colors import LinearSegmentedColormap

        nx = self.dataset.x_dim.size
        ny = self.dataset.y_dim.size
        SP = np.squeeze(self.dataset.speed_t)
        US = np.squeeze(self.dataset.ut_velocity / SP)
        VS = np.squeeze(self.dataset.vt_velocity / SP)

        mask = self.dataset.bottom_level != 0
        p = np.ma.masked_where(mask == 0, SP)
        u = np.ma.masked_where(mask[0::Np, 0::Np] == 0, US[0::Np, 0::Np])
        v = np.ma.masked_where(mask[0::Np, 0::Np] == 0, VS[0::Np, 0::Np])
        # x=nemo_t.dataset.longitude[0::Np,0::Np]
        # y=nemo_t.dataset.latitude[0::Np,0::Np]
        X = self.dataset.longitude
        Y = self.dataset.latitude
        # create a light colour map
        N_colours = int(Vmax * 100)
        n_c = 2
        cmap0 = cm.get_cmap("BrBG_r", lut=N_colours + n_c * 2)
        colors = cmap0(np.arange(cmap0.N))
        colors1 = colors[n_c : cmap0.N - n_c]
        cmap1 = LinearSegmentedColormap.from_list("cmap1", colors1, cmap0.N - n_c * 2)

        cmap1.set_bad([0.75, 0.75, 0.75])
        fig = plt.figure()
        fig.clear()
        plt.pcolormesh(p, cmap=cmap1)
        # plt.pcolormesh(X,Y,p,cmap=cmap1)
        x = np.arange(0, nx, Np)
        y = np.arange(0, ny, Np)

        plt.clim([0, Vmax])
        plt.colorbar(orientation="vertical", cmap=cmap1)
        # plt.quiver(x,y,u,v,color=[0.1,0.1,0.1],headwidth=4,scale=50)
        plt.quiver(x, y, u, v, color=[0.1, 0.1, 0.1], headwidth=headwidth, scale=scale)  # ,**kwargs)
        plt.title("Surface Currents " + name)
