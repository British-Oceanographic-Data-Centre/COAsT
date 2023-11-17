""""
This class calculates the monthly hydrographic climatology
"""
import numpy as np
import xarray as xr

from .._utils.logging_util import debug, warn
from ..data.gridded import Gridded
from ..diagnostics.gridded_stratification import GriddedStratification


class GriddedMonthlyHydrographicClimatology(Gridded):
    """
    Calculates the monthly climatology for sss, sst and pea from multi-annual monthly Gridded data.
    Derived fields (sss, sst, pea) are placed into supplied coast.Gridded object.
    """

    def __init__(self, gridded_t, z_max=200.0):
        """
        Assumes monthly values in gridded_t, starting from Jan and multiyear

        Args:
            gridded_t: Input Gridded object.
            z_max: max z for pea integral calculation
        """
        self.gridded_t = gridded_t
        self.dataset = xr.Dataset()
        self.z_max = z_max

    def calc_climatologies(self):
        """
        Calculate the climatologies for SSH, sss and pea.

        Returns:
            gridded_t: Gridded dataset object.
        """

        # calculate a depth mask
        zd_mask, _, _ = self.gridded_t.calculate_vertical_mask(self.z_max)

        ny = self.gridded_t.dataset.dims["y_dim"]
        nx = self.gridded_t.dataset.dims["x_dim"]

        nt = self.gridded_t.dataset.dims["t_dim"]

        sst_monthy_clim = np.zeros((12, ny, nx))
        sss_monthy_clim = np.zeros((12, ny, nx))
        pea_monthy_clim = np.zeros((12, ny, nx))

        try:
            nyear = int(nt / 12)  # hard wired for monthly data starting in Jan
            for iy in range(nyear):
                print("Calc pea", iy)
                it = np.arange((iy) * 12, (iy) * 12 + 12).astype(int)
                for im in range(12):
                    itt = [it[im]]
                    print(itt)
                    gridded_t2 = self.gridded_t.subset_as_copy(t_dim=itt)
                    print("copied", im)
                    pea = GriddedStratification(gridded_t2)
                    pea.calc_pea(gridded_t2, zd_mask)
                    pea_monthy_clim[im, :, :] = pea_monthy_clim[im, :, :] + pea.dataset["pea"].values
            pea_monthy_clim = pea_monthy_clim / nyear
        except Exception as error:
            (warn(f"Unable to perform pea calculation. Please check the error {error}"))
            debug(f"Unable to perform pea calculation. Please check the error {error}")

            print("not possible to calculate pea")

        sst = self.gridded_t.dataset.variables["sst"]
        sss = self.gridded_t.dataset.variables["sss"]

        for im in range(12):
            print("Month", im)
            it = np.arange(im, nt, 12).astype(int)
            sst_monthy_clim[im, :, :] = np.mean(sst[it, :, :], axis=0)
            sss_monthy_clim[im, :, :] = np.mean(sss[it, :, :], axis=0)
        # NBTy[im,:,:]=np.mean(NBT[it,:,:],axis=0)
        # save hard work in netcdf file
        coords = {
            "Months": (("mon_dim"), np.arange(12).astype(int)),
            "latitude": (("y_dim", "x_dim"), self.gridded_t.dataset.latitude.values),
            "longitude": (("y_dim", "x_dim"), self.gridded_t.dataset.longitude.values),
        }
        dims = ["mon_dim", "y_dim", "x_dim"]
        attributes_sst = {"units": "o^C", "standard name": "Conservative Sea Surface Temperature"}
        attributes_sss = {"units": "", "standard name": "Absolute Sea Surface Salinity"}
        attributes_pea = {"units": "Jm^-3", "standard name": "Potential Energy Anomaly to " + str(self.z_max) + "m"}

        self.dataset = self.gridded_t.dataset["sst_monthy_clim"] = xr.DataArray(
            np.squeeze(sst_monthy_clim), coords=coords, dims=dims, attrs=attributes_sst
        )
        self.gridded_t.dataset["sss_monthy_clim"] = xr.DataArray(
            np.squeeze(sss_monthy_clim), coords=coords, dims=dims, attrs=attributes_sss
        )
        self.gridded_t.dataset["pea_monthy_clim"] = xr.DataArray(
            np.squeeze(pea_monthy_clim), coords=coords, dims=dims, attrs=attributes_pea
        )
        self.dataset = self.gridded_t.dataset
