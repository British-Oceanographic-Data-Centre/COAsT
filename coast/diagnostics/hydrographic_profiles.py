import os
import numpy as np
import xarray as xr
import gsw
from typing import List

from ..data.gridded import Gridded
from ..data.profile import Profile
from ..data.index import Indexed
from dask.diagnostics import ProgressBar

#
Re = 6367456 * np.pi / 180


class Hydrographic_Profiles(Indexed):

    ###############################################################################
    def __init__(self, filename="none", datasetnames="none", config="", regionbounds=[]):
        """Reads and manipulates lists of hydrographic profiles.

        Reads and manipulates lists of hydrographic profiles if called with datasetnames and regionbounds,
        extract profiles in these bounds, and if a filenames is provided, saves them there.
        """
        if datasetnames != "none" and len(regionbounds) == 4:
            self.extractprofiles(datasetnames, regionbounds, config)
            if filename != "none":
                self.saveprofiles(filename)

    def extractprofiles(self, datasetnames, regionbounds, config):
        """
        Args:
            datasetnames: list of file names.
            regionbounds: [lon min, lon max, lat min lat max]
            config : a configuration file (optional)
        """
        x_min = regionbounds[0]
        x_max = regionbounds[1]
        y_min = regionbounds[2]
        y_max = regionbounds[3]
        self.profile = Profile(config=config)
        self.profile.read_en4(datasetnames, multiple=True)
        self.profile = self.profile.subset_indices_lonlat_box(lonbounds=[x_min, x_max], latbounds=[y_min, y_max])
        self.profile = self.profile.process_en4()

    ########################################################################################
    def saveprofiles(self, filename):
        """Saves profile and gridded objects to netcdf."""
        filename_profile = filename[:-3] + "_profile.nc"
        filename_gridded = filename[:-3] + "_gridded.nc"

        print("saving Profile data")
        with ProgressBar():
            self.profile.dataset.to_netcdf(filename_profile)
        print("saving gridded data")
        with ProgressBar():
            self.gridded.dataset.to_netcdf(filename_gridded)

    def loadprofiles(self, filename):
        filename_profile = filename[:-3] + "_profile.nc"
        filename_gridded = filename[:-3] + "_gridded.nc"
        self.profile = Profile()
        dataset = xr.load_dataset(filename_profile)
        self.profile.insert_dataset(dataset)
        dataset = xr.load_dataset(filename_gridded)
        self.gridded.dataset = dataset

    ##############################################################################
    def match_to_grid(self, gridded: Gridded, limits: List = [0, 0, 0, 0], rmax: int = 7000) -> None:
        """Match profiles locations to grid, finding 4 nearest neighbours for each profile.

        Args:
            gridded (Gridded): Gridded object.
            limits (List): [jmin,jmax,imin,imax] - Subset to this region.
            rmax (int): 7000 m maxmimum search distance.
        """
        self.gridded = gridded
        if sum(limits) != 0:
            gridded.subset(ydim=range(limits[0], limits[1] + 0), xdim=range(limits[2], limits[3] + 1))
        # keep the grid or subset on the hydrographic profiles object
        gridded.dataset["limits"] = limits
        self.gridded = gridded
        lon_prf = self.profile.dataset.longitude.values
        lat_prf = self.profile.dataset.latitude.values

        # Find 4 nearest neighbours on grid
        j_prf, i_prf, rmin_prf = gridded.find_j_i_list(lat=lat_prf, lon=lon_prf, n_nn=4)

        self.profile.dataset["i_min"] = limits[0]  # reference back to origianl grid
        self.profile.dataset["j_min"] = limits[2]

        i_min = self.profile.dataset.i_min.values
        j_min = self.profile.dataset.j_min.values

        # Sort 4 NN by distance on grid
        ii = np.nonzero(np.isnan(lon_prf))
        i_prf[ii, :] = 0
        j_prf[ii, :] = 0
        ip = np.where(np.logical_or(i_prf[:, 0] != 0, j_prf[:, 0] != 0))[0]
        lon_prf4 = np.repeat(lon_prf[ip, np.newaxis], 4, axis=1).ravel()
        lat_prf4 = np.repeat(lat_prf[ip, np.newaxis], 4, axis=1).ravel()
        r = np.ones(i_prf.shape) * np.nan
        lon_grd = gridded.dataset.longitude.values
        lat_grd = gridded.dataset.latitude.values

        rr = Hydrographic_Profiles.distance_on_grid(
            lat_grd, lon_grd, j_prf[ip, :].ravel(), i_prf[ip, :].ravel(), lat_prf4, lon_prf4
        )
        r[ip, :] = np.reshape(rr, (ip.size, 4))
        # sort by distance
        ii = np.argsort(r, axis=1)
        rmin_prf = np.take_along_axis(r, ii, axis=1)
        i_prf = np.take_along_axis(i_prf, ii, axis=1)
        j_prf = np.take_along_axis(j_prf, ii, axis=1)

        ii = np.nonzero(np.logical_or(np.min(r, axis=1) > rmax, np.isnan(lon_prf)))
        i_prf = i_prf + i_min
        j_prf = j_prf + j_min
        i_prf[ii, :] = 0  # should the be nan?
        j_prf[ii, :] = 0

        self.profile.dataset["i_prf"] = xr.DataArray(i_prf, dims=["id_dim", "4"])
        self.profile.dataset["j_prf"] = xr.DataArray(j_prf, dims=["id_dim", "4"])
        self.profile.dataset["rmin_prf"] = xr.DataArray(rmin_prf, dims=["id_dim", "4"])

    ###############################################################################
    def stratificationmetrics(self, Zmax: int = 200, DZMAX: int = 30) -> None:
        """Calculates various stratification metrics for observed  profiles.

        Currently: PEA, PEAT, SST, SSS, NBT.

        Args:
            Zmax = 200 m maximum depth of integration.
            DZMAX = 30 m depth of surface layer.
        """
        i_prf = self.profile.dataset.i_prf - self.profile.dataset.i_min
        j_prf = self.profile.dataset.j_prf - self.profile.dataset.j_min
        D = self.gridded.dataset.bathymetry  # uses bathymetry from gridded object
        i_prf = np.ma.masked_less(i_prf, 0)
        j_prf = np.ma.masked_less(j_prf, 0)

        nprof = self.profile.dataset.dims["id_dim"]
        nz = self.profile.dataset.dims["z_dim"]
        sst = np.ones((nprof)) * np.nan
        sss = np.ones((nprof)) * np.nan
        nbt = np.ones((nprof)) * np.nan
        kbot = np.ones((nprof), dtype=int) * np.nan
        PEA = np.ones((nprof)) * np.nan
        PEAT = np.ones((nprof)) * np.nan
        quart = [0, 0.25, 0.5, 0.75, 1]
        # fix memory issues for very large data sets, if this still needed with xarray?
        if nprof < 1000000:
            npr = nprof
        else:
            npr = int(nprof / 10)

        for ichnk in Hydrographic_Profiles.chunks(range(0, nprof), npr):
            Ichnk = list(ichnk)
            print(min(Ichnk), max(Ichnk))
            tmp = self.profile.dataset.potential_temperature[Ichnk, :].values
            sal = self.profile.dataset.practical_salinity[Ichnk, :].values
            ZZ = -self.profile.dataset.depth[Ichnk, :].values
            lat = self.profile.dataset.latitude[Ichnk].values
            lon = self.profile.dataset.longitude[Ichnk].values
            rmin = self.profile.dataset.rmin_prf[Ichnk, :].values
            nprof = len(Ichnk)
            Zd_mask = np.zeros((nprof, nz))

            ################################################################################
            # define interface layers and DZ associated with Z
            Zw = np.empty_like(ZZ)
            DZ = np.empty_like(ZZ)
            Zw[:, 0] = 0.0
            I = np.arange(0, nz - 1)
            Zw[:, I + 1] = 0.5 * (ZZ[:, I] + ZZ[:, I + 1])
            DZ[:, I] = Zw[:, I] - Zw[:, I + 1]
            DZ[~np.isfinite(DZ)] = 0.0
            ZZ[~np.isfinite(ZZ)] = 0.0
            DP = np.ones((nprof)) * np.nan
            # depth from model
            print("Depth from model")
            for ip in range(nprof):

                DP[ip] = 0.0
                rr = 0.0
                for iS in range(0, 4):
                    if D[j_prf[ip, iS], i_prf[ip, iS]] != 0:
                        DP[ip] = DP[ip] + D[j_prf[ip, iS], i_prf[ip, iS]] / rmin[ip, iS]
                        rr = rr + 1 / rmin[ip, iS]
                if rr != 0.0:
                    DP[ip] = DP[ip] / rr
            print("define good profiles")
            good_profile = np.zeros((nprof))
            sstc = np.ones((nprof)) * np.nan
            sssc = np.ones((nprof)) * np.nan
            nbtc = np.ones((nprof)) * np.nan
            kbot = np.ones((nprof), dtype=int) * np.nan
            T = np.zeros(nz) * np.nan
            S = np.zeros(nz) * np.nan
            Z = np.zeros(nz) * np.nan
            ZW = np.zeros(nz) * np.nan
            DP[DP == 0] = np.nan

            for ip in range(nprof):

                Dp = DP[ip]
                T[:] = tmp[ip, :]
                S[:] = sal[ip, :]
                # Z always -ve downwards
                Z[:] = -np.abs(ZZ[ip, :])
                ZW[:] = -np.abs(Zw[ip, :])
                I = np.nonzero(np.isfinite(T))[0]

                if np.size(I) > 0 and np.isfinite(Dp):
                    kbot[ip] = np.max(I)

                    # SST
                    if -Z[np.min(I)] < np.min([DZMAX, 0.25 * Dp]):
                        sstc[ip] = T[np.min(I)]
                    # SSS
                    if -Z[np.min(I)] < np.min([DZMAX, 0.25 * Dp]):
                        sssc[ip] = S[np.min(I)]
                    # Near bototm or ~Zmax temp.
                    if Dp < Zmax:
                        if Dp + Z[int(kbot[ip])] < np.min([DZMAX, 0.25 * Dp]):
                            nbtc[ip] = T[np.max(I)]
                    elif kbot[ip] == nz - 1:
                        nbtc[ip] = T[int(kbot[ip])]
                    elif Z[int(kbot[ip])] < -Zmax and np.size(np.nonzero(Z[I] > -Zmax)) != 0:
                        k = np.max(np.nonzero(Z[I] > -Zmax)[0])
                        k = int(I[k])
                        r = (-Zmax - Z[k]) / (Z[k + 1] - Z[k])
                        nbt[ip] = T[k] * r + T[k + 1] * (1.0 - r)

                    # Depth mask
                    Zd_mask[ip, 0 : int(kbot[ip])] = 1
                    Imax = np.max(np.nonzero(ZW > -Zmax)[0])  # note ZW index
                    if Imax < kbot[ip]:
                        Zd_mask[ip, Imax:nz] = 0
                        Zd_mask[ip, Imax] = (ZW[Imax] - (-Zmax)) / (ZW[Imax] - ZW[Imax + 1])
                        if Zd_mask[ip, Imax - 1] < 0 or Zd_mask[ip, Imax - 1] > 1:
                            print("error", ip, Zd_mask[ip, Imax - 1]), Imax, kbot[ip]
                    # find good profiles

                    DD = np.min([Dp, Zmax])
                    good_profile[ip] = 1
                    for iq in range(len(quart) - 1):
                        I = np.nonzero(
                            np.all(np.concatenate(([Z <= -DD * quart[iq]], [Z >= -DD * quart[iq + 1]]), axis=0), axis=0)
                        )

                        if np.size(I) == 0:
                            good_profile[ip] = 0
                        elif ~(np.any((np.isfinite(S[I]))) and np.any((np.isfinite(S[I])))):
                            good_profile[ip] = 0
                    ###

                    T = Hydrographic_Profiles.fillholes(T)
                    S = Hydrographic_Profiles.fillholes(S)
                    tmp[ip, :] = T
                    sal[ip, :] = S

            ###############################################################################
            print("Calculate metrics")
            metrics = Hydrographic_Profiles.profile_metrics(tmp, sal, ZZ, DZ, Zd_mask, lon, lat)

            PEAc = metrics["PEA"]
            PEATc = metrics["PEAT"]
            PEAc[good_profile == 0] = np.nan
            PEATc[good_profile == 0] = np.nan
            sst[Ichnk] = sstc
            sss[Ichnk] = sssc
            nbt[Ichnk] = nbtc
            PEA[Ichnk] = PEAc
            PEAT[Ichnk] = PEATc
            # Next chunk

        DT = sst - nbt
        self.profile.dataset["PEA"] = xr.DataArray(PEA, dims=["id_dim"])
        self.profile.dataset["PEAT"] = xr.DataArray(PEAT, dims=["id_dim"])
        self.profile.dataset["SST"] = xr.DataArray(sst, dims=["id_dim"])
        self.profile.dataset["SSS"] = xr.DataArray(sss, dims=["id_dim"])
        self.profile.dataset["NBT"] = xr.DataArray(nbt, dims=["id_dim"])
        self.profile.dataset["DT"] = xr.DataArray(DT, dims=["id_dim"])

    def grid_hydro_mnth(self):
        i_prf = self.profile.dataset.i_prf.values[:, 0]
        j_prf = self.profile.dataset.j_prf.values[:, 0]
        varnames = ["SST", "SSS", "PEA", "PEAT", "DT", "NBT"]
        for varname in varnames:
            print("Gridding", varname)
            mnth = self.profile.dataset.time.values.astype("datetime64[M]").astype(int) % 12 + 1
            var, nvar = Hydrographic_Profiles.grid_vars_mnth(self, varname, i_prf, j_prf, mnth)
            self.gridded.dataset[varname] = xr.DataArray(var, dims=["12", "y_dim", "x_dim"])
            self.gridded.dataset["n" + varname] = xr.DataArray(nvar, dims=["12", "y_dim", "x_dim"])

    ###############################################################################
    @staticmethod
    def makefilenames(path, dataset, yr_start, yr_stop):
        if dataset == "EN4":
            datasetnames = []
            for yr in range(yr_start, yr_stop + 1):
                for im in range(1, 13):
                    YR = str(yr)
                    IM = str(im)
                    if im < 10:
                        IM = "0" + IM
                    name = os.path.join(path, f"EN.4.2.1.f.profiles.l09.{YR}{IM}.nc")
                    datasetnames.append(name)
            return datasetnames
        print("Data set not coded")

    # Functions
    ###############################################################################
    # Functions for match to grid
    @staticmethod
    def subsetgrid(var_dom, limits):
        i_min = limits[0]
        i_max = limits[1]
        j_min = limits[2]
        j_max = limits[3]
        if i_max > i_min:
            return var_dom[i_min : i_max + 1, j_min : j_max + 1]
        # special case for wrap-around
        gvar1 = var_dom[i_min:, j_min : j_max + 1]
        gvar2 = var_dom[:i_max, j_min : j_max + 1]
        var_dom = np.concatenate((gvar1, gvar2), axis=0)
        return var_dom

    ###############################################################################
    ###########################################
    def distance_on_grid(Y, X, jpts, ipts, Ypts, Xpts):
        DX = (Xpts - X[jpts, ipts]) * Re * np.cos(Ypts * np.pi / 180.0)
        DY = (Ypts - Y[jpts, ipts]) * Re
        r = np.sqrt(DX**2 + DY**2)
        return r

    ###############################################################################
    # Functions for stratification metrics
    @staticmethod
    def fillholes(Y):
        YY = np.ones(np.shape(Y))
        YY[:] = Y
        I = np.nonzero(np.isfinite(YY))
        N = len(YY)

        if np.size(I) > 0:
            if not np.isfinite(YY[0]):
                YY[0 : np.min(I) + 1] = YY[np.min(I)]

            if ~np.isfinite(YY[N - 1]):
                YY[np.max(I) : N] = YY[np.max(I)]
            I = np.array(np.nonzero(~np.isfinite(YY)))
            YY[I] = 0.5 * (YY[I - 1] + YY[I + 1])
            YYp = YY[0]
            ip = 0
            for i in range(N):
                if np.isfinite(YY[i]):
                    YYp = YY[i]
                    ip = i
                else:
                    j = i
                    while ~np.isfinite(YY[j]):
                        j = j + 1
                    Jp = np.arange(ip + 1, j - 1 + 1)

                    pT = np.arange(1.0, (j - ip - 1.0) + 1.0) / (j - ip)
                    YY[Jp] = YYp + (YY[j] - YYp) * pT
        return YY

    ###########################################
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    ###########################################
    @staticmethod
    def profile_metrics(tmp, sal, Z, DZ, Zd_mask, lon, lat):

        metrics = {}
        g = 9.81
        DD = np.sum(DZ * Zd_mask, axis=1)
        nz = Z.shape[1]
        lat = np.repeat(lat[:, np.newaxis], nz, axis=1)
        lon = np.repeat(lon[:, np.newaxis], nz, axis=1)
        pressure_absolute = gsw.p_from_z(Z, lat)
        salinity_absolute = gsw.SA_from_SP(sal, pressure_absolute, lon, lat)
        temp_conservative = gsw.CT_from_pt(salinity_absolute, tmp)
        rho = np.ma.masked_invalid(gsw.rho(salinity_absolute, temp_conservative, 0.0))

        Tbar = np.sum(temp_conservative * DZ * Zd_mask, axis=1) / DD
        Sbar = np.sum(salinity_absolute * DZ * Zd_mask, axis=1) / DD

        rhobar = np.ma.masked_invalid(gsw.rho(Sbar, Tbar, 0.0))
        rhobar_2d = np.repeat(rhobar[:, np.newaxis], nz, axis=1)
        Sbar_2d = np.repeat(Sbar[:, np.newaxis], nz, axis=1)
        rhoT = np.ma.masked_invalid(gsw.rho(Sbar_2d, temp_conservative, 0.0))  # density with constant salinity

        PEA = -np.sum(Z * (rho - rhobar_2d) * DZ * Zd_mask, axis=1) * g / DD
        PEAT = -np.sum(Z * (rhoT - rhobar_2d) * DZ * Zd_mask, axis=1) * g / DD

        metrics["PEA"] = PEA
        metrics["PEAT"] = PEAT

        return metrics

    ###########################################

    def grid_vars_mnth(self, var, i_var, j_var, mnth_var):
        VAR = self.profile.dataset[var].values
        nx = self.gridded.dataset.dims["x_dim"]
        ny = self.gridded.dataset.dims["y_dim"]

        Ig = np.nonzero(np.isfinite(VAR))[0]

        var = VAR[Ig]
        VAR_g = np.zeros((12, ny, nx))
        nVAR_g = np.zeros((12, ny, nx))
        for ip in range(0, np.size(Ig)):
            i = i_var[Ig[ip]]
            j = j_var[Ig[ip]]
            im = int(mnth_var[Ig[ip]]) - 1

            VAR_g[im, j, i] = VAR_g[im, j, i] + var[ip]
            nVAR_g[im, j, i] = nVAR_g[im, j, i] + 1

        VAR_g = VAR_g / nVAR_g
        return VAR_g, nVAR_g
