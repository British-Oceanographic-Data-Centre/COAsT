from ..data.gridded import Gridded
from ..diagnostics.internal_tide import InternalTide
import numpy as np
import xarray as xr


class Annual_Climatology(Gridded):
    """
    Calculates a mean annual cycle from multi-annual monthly data
    Because it calculates dervied properties (e.g PEA), data must be loaded.
    Currently hardwired to calculate SST, SSS and PEA, placing these in the Gridded Objected
    """

    def __init__(self, gridded_t, gridded_t_out, Zmax=200.0):

        # calculate a depth mask
        Zd_mask, kmax, Ikmax = gridded_t.calculate_vertical_mask(Zmax)

        ny = gridded_t.dataset.dims["y_dim"]
        nx = gridded_t.dataset.dims["x_dim"]

        nt = gridded_t.dataset.dims["t_dim"]

        SSTy = np.zeros((12, ny, nx))
        SSSy = np.zeros((12, ny, nx))
        PEAy = np.zeros((12, ny, nx))
        # NBTy=np.zeros((12,ny,nx)) #will add near bed temperature later

        PEAy = np.zeros((12, ny, nx))

        nyear = int(nt / 12)  # hard wired for monthly data starting in Jan
        for iy in range(nyear):
            print("Calc PEA", iy)
            it = np.arange((iy) * 12, (iy) * 12 + 12).astype(int)
            for im in range(12):
                itt = [it[im]]
                print(itt)
                gridded_t2 = gridded_t.subset_as_copy(t_dim=itt)
                print("copied", im)
                PEA = InternalTide(gridded_t2, gridded_t2)
                PEA.calc_pea(gridded_t2, Zd_mask)
                PEAy[im, :, :] = PEAy[im, :, :] + PEA.dataset["PEA"].values
        PEAy = PEAy / nyear

        # need to find efficient method for bottom temperature
        # NBT=np.zeros((nt,ny,nx))
        # for it in range(nt):
        #    NBT[it,:,:]=np.reshape(tmp[it,:,:,:].values.ravel()[Ikmax],(ny,nx))
        SST = gridded_t.dataset.variables["temperature"][:, 0, :, :]
        SSS = gridded_t.dataset.variables["salinity"][:, 0, :, :]

        for im in range(12):
            print("Month", im)
            it = np.arange(im, nt, 12).astype(int)
            SSTy[im, :, :] = np.mean(SST[it, :, :], axis=0)
            SSSy[im, :, :] = np.mean(SSS[it, :, :], axis=0)
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
        attributes_PEA = {"units": "Jm^-3", "standard name": "Potential Energy Anomaly to 200m"}
        gridded_t_out.dataset["SSTy"] = xr.DataArray(np.squeeze(SSTy), coords=coords, dims=dims, attrs=attributes_SST)
        gridded_t_out.dataset["SSSy"] = xr.DataArray(np.squeeze(SSSy), coords=coords, dims=dims, attrs=attributes_SSS)
        gridded_t_out.dataset["PEAy"] = xr.DataArray(np.squeeze(PEAy), coords=coords, dims=dims, attrs=attributes_PEA)


#%%
