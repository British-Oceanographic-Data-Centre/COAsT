from ..data.gridded import Gridded
from ..diagnostics.gridded_stratification import GriddedStratification
import numpy as np
import xarray as xr


class GriddedMonthlyHydrographicClimatology(Gridded):
    """
    Calculates the monthly climatology for SSS, SST and PEA from multi-annual monthly Gridded data.
    Derived fields (SSS, SST, PEA) are placed into supplied coast.Gridded object.
    """

    def __init__(self, gridded_t, gridded_t_out, Zmax=200.0):
        """
        Assumes monthly values in gridded_t, starting from Jan and multiyear

        Args:
            gridded_t: Input Gridded object.
            gridded_t: Target Gridded object
            Zmax: Max z for PEA integral calculation
        """

        # calculate a depth mask
        Zd_mask, _, _ = gridded_t.calculate_vertical_mask(Zmax)

        ny = gridded_t.dataset.dims["y_dim"]
        nx = gridded_t.dataset.dims["x_dim"]

        nt = gridded_t.dataset.dims["t_dim"]

        SST_monthy_clim = np.zeros((12, ny, nx))
        SSS_monthy_clim = np.zeros((12, ny, nx))
        PEA_monthy_clim = np.zeros((12, ny, nx))
        # NBTy=np.zeros((12,ny,nx)) #will add near bed temperature later

        PEA_monthy_clim = np.zeros((12, ny, nx))

        nyear = int(nt / 12)  # hard wired for monthly data starting in Jan
        for iy in range(nyear):
            print("Calc PEA", iy)
            it = np.arange((iy) * 12, (iy) * 12 + 12).astype(int)
            for im in range(12):
                itt = [it[im]]
                print(itt)
                gridded_t2 = gridded_t.subset_as_copy(t_dim=itt)
                print("copied", im)
                PEA = GriddedStratification(gridded_t2, gridded_t2)
                PEA.calc_pea(gridded_t2, Zd_mask)
                PEA_monthy_clim[im, :, :] = PEA_monthy_clim[im, :, :] + PEA.dataset["PEA"].values
        PEA_monthy_clim = PEA_monthy_clim / nyear

        # need to find efficient method for bottom temperature
        # NBT=np.zeros((nt,ny,nx))
        # for it in range(nt):
        #    NBT[it,:,:]=np.reshape(tmp[it,:,:,:].values.ravel()[Ikmax],(ny,nx))
        SST = gridded_t.dataset.variables["temperature"][:, 0, :, :]
        SSS = gridded_t.dataset.variables["salinity"][:, 0, :, :]

        for im in range(12):
            print("Month", im)
            it = np.arange(im, nt, 12).astype(int)
            SST_monthy_clim[im, :, :] = np.mean(SST[it, :, :], axis=0)
            SSS_monthy_clim[im, :, :] = np.mean(SSS[it, :, :], axis=0)
        # NBTy[im,:,:]=np.mean(NBT[it,:,:],axis=0)
        # save hard work in netcdf file
        coords = {
            "Months": (("mon_dim"), np.arange(12).astype(int)),
            "latitude": (("y_dim", "x_dim"), gridded_t.dataset.latitude.values),
            "longitude": (("y_dim", "x_dim"), gridded_t.dataset.longitude.values),
        }
        dims = ["mon_dim", "y_dim", "x_dim"]
        attributes_SST = {"units": "o^C", "standard name": "Conservative Sea Surface Temperature"}
        attributes_SSS = {"units": "", "standard name": "Absolute Sea Surface Salinity"}
        attributes_PEA = {"units": "Jm^-3", "standard name": "Potential Energy Anomaly to " + str(Zmax) + "m"}
        gridded_t_out.dataset["SST_monthy_clim"] = xr.DataArray(
            np.squeeze(SST_monthy_clim), coords=coords, dims=dims, attrs=attributes_SST
        )
        gridded_t_out.dataset["SSS_monthy_clim"] = xr.DataArray(
            np.squeeze(SSS_monthy_clim), coords=coords, dims=dims, attrs=attributes_SSS
        )
        gridded_t_out.dataset["PEA_monthy_clim"] = xr.DataArray(
            np.squeeze(PEA_monthy_clim), coords=coords, dims=dims, attrs=attributes_PEA
        )
