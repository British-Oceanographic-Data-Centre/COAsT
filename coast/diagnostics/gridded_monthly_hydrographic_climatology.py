""" "
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

    def __init__(self, gridded_t, z_max=200.0, variables=['pea','sst','sss']):
        """
        Assumes monthly values in gridded_t, starting from Jan and multiyear

        Args:
            gridded_t: Input Gridded object.
            z_max: max z for pea integral calculation
        """
        self.gridded_t = gridded_t
        self.dataset = xr.Dataset()
        self.z_max = z_max
        self.variables = variables

    def calc_climatologies(self):
        """
                Calculate the climatologies for SSH, sss and pea.

                Returns:
        #            gridded_t: Gridded dataset object.
                     dataset:  Gridded dataset object containing monthly climatologies
        """

        # calculate a depth mask
        zd_mask, _, _ = self.gridded_t.calculate_vertical_mask(self.z_max)

        ny = self.gridded_t.dataset.sizes["y_dim"]
        nx = self.gridded_t.dataset.sizes["x_dim"]
        nt = self.gridded_t.dataset.sizes["t_dim"]
        monthly_clim={}
        variables = self.variables
        for var in variables:
            monthly_clim[var] = np.zeros((12, ny, nx))
        for var in variables:
            if var == 'pea':
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
                            monthly_clim['pea'][im, :, :] = monthly_clim['pea'][im, :, :] + pea.dataset["PEA"].values
                    monthly_clim['pea'] = monthly_clim['pea'] / nyear

                except Exception as error:
                    (warn(f"Unable to perform pea calculation. Please check the error {error}"))
                    debug(f"Unable to perform pea calculation. Please check the error {error}")

                    print("not possible to calculate pea")
    #        if "sst" in self.gridded_t.dataset.variables:
    #            sst = self.gridded_t.dataset.variables["sst"]
    #        else:
    #            sst = self.gridded_t.dataset.variables["temperature"].values[:,0,:,:]
    #        if "sss" in self.gridded_t.dataset.variables:
    #            sss = self.gridded_t.dataset.variables["sss"]
    #        else:
    #            sss = self.gridded_t.dataset.variables["salinity"].values[:,0,:,:]

            else:
                if self.gridded_t.dataset.variables[var].ndim == 3:
                    Var = self.gridded_t.dataset.variables[var]
                else:
                    Var = self.gridded_t.dataset.variables[var].values[:,0,:,:]
                for im in range(12):
                    print("Month", var, im)
                    it = np.arange(im, nt, 12).astype(int)
                    monthly_clim[var][im, :, :] = np.mean(Var[it, :, :], axis=0)

        # NBTy[im,:,:]=np.mean(NBT[it,:,:],axis=0)
        # save hard work in netcdf file
        coords = {
            "Months": (("mon_dim"), np.arange(12).astype(int)),
            "latitude": (("y_dim", "x_dim"), self.gridded_t.dataset.latitude.values),
            "longitude": (("y_dim", "x_dim"), self.gridded_t.dataset.longitude.values),
        }
        dims = ["mon_dim", "y_dim", "x_dim"]
        #attributes_sst = {"units": "o^C", "standard name": "Conservative Sea Surface Temperature"}
        #attributes_sss = {"units": "", "standard name": "Absolute Sea Surface Salinity"}
        #attributes_pea = {"units": "Jm^-3", "standard name": "Potential Energy Anomaly to " + str(self.z_max) + "m"}
        # jth this adds the new variables to the full data set, which makes saving difficult, easier just to keep the new variables in seperate object
        #        self.dataset = self.gridded_t.dataset["sst_monthy_clim"] = xr.DataArray(
        #            np.squeeze(sst_monthy_clim), coords=coords, dims=dims, attrs=attributes_sst
        #        )
        #        self.gridded_t.dataset["sss_monthy_clim"] = xr.DataArray(
        #            np.squeeze(sss_monthy_clim), coords=coords, dims=dims, attrs=attributes_sss
        #        )
        #        self.gridded_t.dataset["pea_monthy_clim"] = xr.DataArray(
        #            np.squeeze(pea_monthy_clim), coords=coords, dims=dims, attrs=attributes_pea
        #        )
        #        self.dataset = self.gridded_t.dataset
        for var in variables:
            if var == 'pea':
                attributes = {"units": "Jm^-3", "standard name": "Potential Energy Anomaly to " + str(self.z_max) + "m"}
            else:
                attributes = self.gridded_t.dataset.variables[var].attrs
            self.dataset[f"{var}_monthly_clim"] = xr.DataArray(
                np.squeeze(monthly_clim[var]), coords=coords, dims=dims, attrs=attributes
                )

 #       self.dataset["sss_monthy_clim"] = xr.DataArray(
 #           np.squeeze(sss_monthy_clim), coords=coords, dims=dims, attrs=attributes_sss
 #       )
 #       self.dataset["pea_monthy_clim"] = xr.DataArray(
 #           np.squeeze(pea_monthy_clim), coords=coords, dims=dims, attrs=attributes_pea
 #       )
