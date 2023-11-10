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
from .._utils.logging_util import debug


class CurrentsOnT(Gridded):
    """
    Methods for currents co-located on T-points

    Use like coast.Gridded() with a domain_cfg file name to create gridded object

    """

    def __init(self, fn_domain=None, config=None, **kwargs):
        gridded = Gridded(fn_domain=fn_domain, config=config, **kwargs)
        self.dataset = gridded.dataset

    def currents_on_t(self, gridded_u, gridded_v):
        """
        Adds co-located velocity components and speed to CurrentsOnT object

        Avoids use of indices in the hope that it will be more parallelisable. However in index space the mappings
        take this form:
        u_on_t_points[:, :, :, 1:] = 0.5 * (
            gridded_u.dataset.u_velocity.values[:, :, :, 1:] + gridded_u.dataset.u_velocity.values[:, :, :, :-1]
            )

        v_on_t_points[:, :, 1:, :] = 0.5 * (
            gridded_v.dataset.v_velocity.values[:, :, 1:, :] + gridded_v.dataset.v_velocity.values[:, :, :-1, :]
            )

        For arrays of shape and order: (z_dim, t_dim, y_dim, x_dim)

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
        ds_u = self.dataset.copy(deep=True)
        # U velocity on T-points
        ds_u["ut_velocity"] = 0.5 * (gridded_u.dataset.u_velocity.shift(x_dim=1) + gridded_u.dataset.u_velocity)
        # replace wrapped (1st) longitude coord with zero
        _, _lon = xr.broadcast(gridded_u.dataset.u_velocity, gridded_u.dataset.longitude)
        ds_u["ut_velocity"] = ds_u["ut_velocity"].where(
            _lon != _lon.isel(x_dim=0), 0
        )  # keep values except where lon(x_dim=0)
        del _, _lon
        ds_u.coords["latitude"] = self.dataset.latitude
        ds_u.coords["longitude"] = self.dataset.longitude
        ds_u.coords["depth_0"] = self.dataset.depth_0
        try:
            self.dataset["ut_velocity"] = ds_u.ut_velocity.drop_vars("depthu")
        except:
            self.dataset["ut_velocity"] = ds_u.ut_velocity
            debug("Did not find depthu variable to drop - to avoid conflicts in z_dim dimension")

        # V velocity on T-points
        ds_v = self.dataset.copy(deep=True)

        ds_v["vt_velocity"] = 0.5 * (gridded_v.dataset.v_velocity.shift(y_dim=1) + gridded_v.dataset.v_velocity)
        # replace wrapped (1st) latitude coord with zero
        _, _lat = xr.broadcast(gridded_v.dataset.v_velocity, gridded_v.dataset.latitude)
        ds_v["vt_velocity"] = ds_v["vt_velocity"].where(
            _lat != _lat.isel(y_dim=0), 0
        )  # keep values except where lat(y_dim=0)
        del _, _lat
        ds_v.coords["latitude"] = self.dataset.latitude
        ds_v.coords["longitude"] = self.dataset.longitude
        ds_v.coords["depth_0"] = self.dataset.depth_0
        try:
            self.dataset["vt_velocity"] = ds_v.vt_velocity.drop_vars("depthv")
        except:
            self.dataset["vt_velocity"] = ds_v.vt_velocity
            debug("Did not find depthv variable to drop - to avoid conflicts in z_dim dimension")

        # Speed on T-points
        self.dataset["speed_t"] = np.sqrt(
            np.square(self.dataset["ut_velocity"]) + np.square(self.dataset["vt_velocity"])
        )

    def quick_plot(self, name, Vmax=0.16, Np=3, headwidth=4, scale=50, time_value=None, **kwargs):
        """
        plot surface circulation
        direction: unit vector
        magnitude: shaded
        """
        # %%
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        nx = self.dataset.x_dim.size
        ny = self.dataset.y_dim.size

        if time_value == None:
            SP = np.squeeze(self.dataset.speed_t)
            US = np.squeeze(self.dataset.ut_velocity / SP)
            VS = np.squeeze(self.dataset.vt_velocity / SP)
        else:
            SP = np.squeeze(self.dataset.speed_t[time_value, :, :])
            US = np.squeeze(self.dataset.ut_velocity[time_value, :, :] / SP)
            VS = np.squeeze(self.dataset.vt_velocity[time_value, :, :] / SP)
        mask = self.dataset.bottom_level != 0
        p = np.ma.masked_where(mask == 0, SP)
        u = np.ma.masked_where(mask[0::Np, 0::Np] == 0, US[0::Np, 0::Np])
        v = np.ma.masked_where(mask[0::Np, 0::Np] == 0, VS[0::Np, 0::Np])
        # x=nemo_t.dataset.longitude[0::Np,0::Np]
        # y=nemo_t.dataset.latitude[0::Np,0::Np]
        X = self.dataset.longitude
        Y = self.dataset.latitude
        # create a light colour map
        n_colours = int(Vmax * 100)
        n_c = 2
        cmap0 = plt.get_cmap("BrBG_r", lut=n_colours + n_c * 2)
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
        return fig
