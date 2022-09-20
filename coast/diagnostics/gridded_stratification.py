from ..data.gridded import Gridded
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import copy
from .._utils.logging_util import get_slug, debug


class GriddedStratification(Gridded):  # TODO All abstract methods should be implemented
    """
    Object for handling and storing necessary information, methods and outputs
    for calculation of stratification diagnostics.

    Herein the depth moments of stratification are used as proxies for
    pycnocline depth (as the first  moment of stratification), and pycnocline
    thickness  (as the 2nd moment of stratification).
    This approximation improves towards the limit of a two-layer fluid.

    For stratification that is not nearly two-layer, the pycnocline
    thickness appears large and this method for identifying the pycnocline
    depth is less reliable.

    Parameters
    ----------
        gridded_t : xr.Dataset
            Gridded object on t-points.
        gridded_w : xr.Dataset, optional
            Gridded object on w-points.

    Example basic usage:
    -------------------
        # Create Internal tide diagnostics object
        strat_obj = GriddedStratification(gridded_t, gridded_w) # For Gridded objects on t and w-pts
        strat_obj.construct_pycnocline_vars( gridded_t, gridded_w )
        # Make maps of pycnocline thickness and depth
        strat_obj.quick_plot()
    """

    def __init__(self, gridded_t: xr.Dataset):
        # TODO Super __init__ should be called at some point
        debug(f"Creating new {get_slug(self)}")
        self.dataset = xr.Dataset()
        self.filename_domain = gridded_t.filename_domain

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in
        # z_dim, ydim, xdim
        self.nt = gridded_t.dataset.dims["t_dim"]
        self.nz = gridded_t.dataset.dims["z_dim"]
        self.ny = gridded_t.dataset.dims["y_dim"]
        self.nx = gridded_t.dataset.dims["x_dim"]
        debug(f"Initialised {get_slug(self)}")

    def construct_pycnocline_vars(self, gridded_t: Gridded, gridded_w: Gridded, strat_thres=-0.01):
        """
        Computes depth moments of stratification. Under the assumption that the
        stratification approximately represents a two-layer fluid, these can be
        interpreted as pycnocline depths and thicknesses. They are computed on
        w-points.

        1st moment of stratification: \int z.strat dz / \int strat dz
            In the limit of a two layer fluid this is equivalent to the
        pycnocline depth, or z_d (units: metres)

        2nd moment of stratification: \sqrt{\int (z-z_d)^2 strat dz / \int strat dz}
            where strat = d(density)/dz
            In the limit of a two layer fluid this is equivatlent to the
        pycnocline thickness, or z_t (units: metres)

        Parameters
        ----------
        gridded_t : Gridded
            Gridded object on t-points.
        gridded_w : Gridded, optional
            Gridded object on w-points.
        strat_thres: float - Optional
            limiting stratification (rho_dz < 0) to trigger masking of mixed waters

        Output
        ------
        self.dataset.strat_1st_mom - (t,y,x) pycnocline depth
        self.dataset.strat_2nd_mom - (t,y,x) pycnocline thickness
        self.dataset.strat_1st_mom_masked - (t,y,x) pycnocline depth, masked
                in weakly stratified water beyond strat_thres
        self.dataset.strat_2nd_mom_masked - (t,y,x) pycnocline thickness, masked
                in weakly stratified water beyond strat_thres
        self.dataset.mask - (t,y,x) [1/0] stratified/unstrafied
                water column according to strat_thres not being met anywhere
                in the column

        Returns
        -------
        None.

        Example Usage
        -------------
        # load some example data
        dn_files = "./example_files/"
        dn_fig = 'unit_testing/figures/'
        fn_nemo_grid_t_dat = 'nemo_data_T_grid_Aug2015.nc'
        fn_nemo_dom = 'coast_example_nemo_domain.nc'
        gridded_t = coast.Gridded(dn_files + fn_nemo_grid_t_dat,
                     dn_files + fn_nemo_dom, grid_ref='t-grid')
        # create an empty w-grid object, to store stratification
        gridded_w = coast.Gridded( fn_domain = dn_files + fn_nemo_dom,
                           grid_ref='w-grid')

        # initialise GriddedStratification object
        strat = coast.GriddedStratification(gridded_t, gridded_w)
        # Construct pycnocline variables: depth and thickness
        strat.construct_pycnocline_vars( gridded_t, gridded_w )
        # Plot pycnocline depth and thickness
        strat.quickplot()

        """

        debug(f"Constructing pycnocline variables for {get_slug(self)}")
        # Construct in-situ density if not already done
        if not hasattr(gridded_t.dataset, "density"):
            gridded_t.construct_density(eos="EOS10")

        # Construct stratification if not already done. t-pts --> w-pts
        if not hasattr(gridded_w.dataset, "rho_dz"):
            gridded_w = gridded_t.differentiate("density", dim="z_dim", out_var_str="rho_dz", out_obj=gridded_w)

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in
        # z_dim, ydim, xdim
        nt = gridded_t.dataset.dims["t_dim"]
        # nz = gridded_t.dataset.dims['z_dim']
        ny = gridded_t.dataset.dims["y_dim"]
        nx = gridded_t.dataset.dims["x_dim"]

        # Create a mask for weakly stratified waters
        # Preprocess stratification
        strat = copy.copy(gridded_w.dataset.rho_dz)  # (t_dim, z_dim, ydim, xdim). w-pts.
        # Ensure surface value is 0
        strat[:, 0, :, :] = 0
        # Ensure bed value is 0
        strat[:, -1, :, :] = 0
        # mask out the Nan values
        strat = strat.where(~np.isnan(gridded_w.dataset.rho_dz), drop=False)
        # create mask with a stratification threshold
        strat_m = gridded_w.dataset.latitude * 0 + 1  # create a stratification mask: [1/0] = strat/un-strat
        strat_m = strat_m.where(strat.min(dim="z_dim").squeeze() < strat_thres, 0, drop=False)
        strat_m = strat_m.transpose("t_dim", "y_dim", "x_dim", transpose_coords=True)

        # Compute statification variables
        # initialise pycnocline variables
        pycnocline_depth = np.zeros((nt, ny, nx))  # pycnocline depth
        zt = np.zeros((nt, ny, nx))  # pycnocline thickness

        # Construct intermediate variables
        # Broadcast to fill out missing (time) dimensions in grid data
        _, depth_0_4d = xr.broadcast(strat, gridded_w.dataset.depth_0)
        _, e3_0_4d = xr.broadcast(strat, gridded_w.dataset.e3_0.squeeze())

        # integrate strat over depth
        intN2 = (strat * e3_0_4d).sum(
            dim="z_dim", skipna=True
        )  # TODO Can someone sciencey give me the proper name for this?
        # integrate (depth * strat) over depth
        intzN2 = (strat * e3_0_4d * depth_0_4d).sum(
            dim="z_dim", skipna=True
        )  # TODO Can someone sciencey give me the proper name for this?

        # compute pycnocline depth
        pycnocline_depth = intzN2 / intN2  # pycnocline depth

        # compute pycnocline thickness
        intz2N2 = (np.square(depth_0_4d - pycnocline_depth) * e3_0_4d * strat).sum(
            dim="z_dim", skipna=True
        )  # TODO Can someone sciencey give me the proper name for this?
        zt = np.sqrt(intz2N2 / intN2)  # pycnocline thickness

        # Define xarray attributes
        coords = {
            "time": ("t_dim", gridded_t.dataset.time.values),
            "latitude": (("y_dim", "x_dim"), gridded_t.dataset.latitude.values),
            "longitude": (("y_dim", "x_dim"), gridded_t.dataset.longitude.values),
        }
        dims = ["t_dim", "y_dim", "x_dim"]

        # Save a xarray objects
        self.dataset["strat_2nd_mom"] = xr.DataArray(zt, coords=coords, dims=dims)
        self.dataset.strat_2nd_mom.attrs["units"] = "m"
        self.dataset.strat_2nd_mom.attrs["standard_name"] = "pycnocline thickness"
        self.dataset.strat_2nd_mom.attrs["long_name"] = "Second depth moment of stratification"

        self.dataset["strat_1st_mom"] = xr.DataArray(pycnocline_depth, coords=coords, dims=dims)
        self.dataset.strat_1st_mom.attrs["units"] = "m"
        self.dataset.strat_1st_mom.attrs["standard_name"] = "pycnocline depth"
        self.dataset.strat_1st_mom.attrs["long_name"] = "First depth moment of stratification"

        # Mask pycnocline variables in weak stratification
        zd_m = pycnocline_depth.where(strat_m > 0)
        zt_m = zt.where(strat_m > 0)

        self.dataset["mask"] = xr.DataArray(strat_m, coords=coords, dims=dims)

        self.dataset["strat_2nd_mom_masked"] = xr.DataArray(zt_m, coords=coords, dims=dims)
        self.dataset.strat_2nd_mom_masked.attrs["units"] = "m"
        self.dataset.strat_2nd_mom_masked.attrs["standard_name"] = "masked pycnocline thickness"
        self.dataset.strat_2nd_mom_masked.attrs[
            "long_name"
        ] = "Second depth moment of stratification, masked in weak stratification"

        self.dataset["strat_1st_mom_masked"] = xr.DataArray(zd_m, coords=coords, dims=dims)
        self.dataset.strat_1st_mom_masked.attrs["units"] = "m"
        self.dataset.strat_1st_mom_masked.attrs["standard_name"] = "masked pycnocline depth"
        self.dataset.strat_1st_mom_masked.attrs[
            "long_name"
        ] = "First depth moment of stratification, masked in weak stratification"

        # Inherit horizontal grid information from gridded_w
        self.dataset["e1"] = xr.DataArray(
            gridded_w.dataset.e1,
            coords={
                "latitude": (("y_dim", "x_dim"), gridded_t.dataset.latitude.values),
                "longitude": (("y_dim", "x_dim"), gridded_t.dataset.longitude.values),
            },
            dims=["y_dim", "x_dim"],
        )
        self.dataset["e2"] = xr.DataArray(
            gridded_w.dataset.e2,
            coords={
                "latitude": (("y_dim", "x_dim"), gridded_t.dataset.latitude.values),
                "longitude": (("y_dim", "x_dim"), gridded_t.dataset.longitude.values),
            },
            dims=["y_dim", "x_dim"],
        )

    def calc_pea(self, gridded_t: xr.Dataset, Zd_mask):
        """
        Calculates Potential Energy Anomaly

        The density and depth averaged density can be supplied within gridded_t as "density" and
        "density_bar" DataArrays, respectively. If they are not supplied they will be calculated.
        "density_bar" is calculated using depth averages of temperature and salinity.

        Example Usage: PEA in upper 200m
        --------------------------------
        # load some example data. E.g.
        root = "~/work/coast/"
        dn_files = root + "./example_files/"
        fn_nemo_grid_t_dat = dn_files + "nemo_data_T_grid_Aug2015.nc"
        fn_nemo_dom = dn_files + "coast_example_nemo_domain.nc"
        config_t = root + "./config/example_nemo_grid_t.json"
        dn_fig = 'unit_testing/figures/'
        gridded_t = coast.Gridded(fn_nemo_grid_t_dat, fn_nemo_dom, config=config_t)
        Zd_mask,kmax,Ikmax=gridded_t.calculate_vertical_mask(200.)
        strat=coast.GriddedStratification(gridded_t)
        strat.calc_pea(gridded_t,Zd_mask)
        strat.quick_plot('PEA')
        """
        # may be duplicated in other branches. Uses the integral of T&S rather than integral of rho approach
        gravity = 9.81
        # Z=gridded_t.dataset.variables['depth_0'].values
        # DZ=gridded_t.dataset.variables['e3_0'].values*Zd_mask
        _, z_4d = xr.broadcast(gridded_t.dataset.salinity, gridded_t.dataset.depth_0)
        _, dz_4d = xr.broadcast(gridded_t.dataset.salinity, gridded_t.dataset.e3_0.squeeze() * Zd_mask)
        height = dz_4d.sum(dim="z_dim", skipna=True)  # water depth or Zmax ,
        #         H=xr.broadcast(gridded_t.dataset.salinity,H)[0]
        #         nt=gridded_t.dataset.dims['t_dim']

        if not "density" in gridded_t.dataset:
            gridded_t.construct_density(CT_AS=True, pot_dens=True)
        if not "density_bar" in gridded_t.dataset:
            gridded_t.construct_density(CT_AS=True, rhobar=True, Zd_mask=Zd_mask, pot_dens=True)
        rho = gridded_t.dataset.variables["density"].values  # density
        rho[np.isnan(rho)] = 0
        rhobar = gridded_t.dataset.variables["density_bar"]  # density with depth-mean T and S

        #         z_axis=0
        #         if len(gridded_t.dataset['density'].shape) == 4:   # includes time as first axis
        #          Z=np.repeat(Z[np.newaxis,:,:,:],nt,axis=0)
        #          DZ=np.repeat(DZ[np.newaxis,:,:,:],nt,axis=0)
        #          DP=np.repeat(DP[np.newaxis,:,:],nt,axis=0)
        #          z_axis=1

        PEA = (z_4d * (rho - rhobar) * dz_4d).sum(dim="z_dim", skipna=True) * gravity / height
        #%%
        #         return PEA
        coords = {
            "time": ("t_dim", gridded_t.dataset.time.values),
            "latitude": (("y_dim", "x_dim"), gridded_t.dataset.latitude.values),
            "longitude": (("y_dim", "x_dim"), gridded_t.dataset.longitude.values),
        }
        dims = ["t_dim", "y_dim", "x_dim"]
        attributes = {"units": "J / m^3", "standard_name": "Potential Energy Anomaly"}
        self.dataset["PEA"] = xr.DataArray(PEA, coords=coords, dims=dims, attrs=attributes)

    def quick_plot(self, var: xr.DataArray = None):
        """

        Map plot for pycnocline depth and thickness variables.

        Parameters
        ----------
        var : xr.DataArray, optional
            Pass variable to plot. The default is None. In which case both
            strat_1st_mom and strat_2nd_mom are plotted.

        Returns
        -------
        None.

        Example Usage
        -------------
        strat.quick_plot( 'strat_1st_mom_masked' )

        """

        debug(f"Generating quick plot for {get_slug(self)}")

        if var is None:
            var_lst = [self.dataset.strat_1st_mom_masked, self.dataset.strat_2nd_mom_masked]
        else:
            var_lst = [self.dataset[var]]

        fig = None
        ax = None
        for var in var_lst:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            plt.pcolormesh(self.dataset.longitude.squeeze(), self.dataset.latitude.squeeze(), var.isel(t_dim=0))
            #               var.mean(dim = 't_dim') )
            # plt.contourf( self.dataset.longitude.squeeze(),
            #               self.dataset.latitude.squeeze(),
            #               var.mean(dim = 't_dim'), levels=(0,10,20,30,40) )
            title_str = (
                self.dataset.time[0].dt.strftime("%d %b %Y: ").values
                + var.attrs["standard_name"]
                + " ("
                + var.attrs["units"]
                + ")"
            )
            plt.title(title_str)
            plt.xlabel("longitude")
            plt.ylabel("latitude")
            plt.clim([0, 50])
            plt.colorbar()
            plt.show()
        return fig, ax
