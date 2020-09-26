from .COAsT import COAsT
import numpy as np
import xarray as xr
import copy
import gsw  # TODO Use this or remove the import
from .logging_util import get_slug, debug


class INTERNALTIDE(COAsT):
    """
    Object for handling and storing necessary information, methods and outputs
    for calculation of internal tide diagnostics.

    Herein the depth moments of stratification are used as proxies for
    pycnocline depth (as the first  moment of stratification), and pycnocline
    thickness  (as the 2nd moment of stratification).
    This approximation improves towards the limit of a two-layer fluid.

    For stratification that is not nearly two-layer, the pycnocline
    thickness appears large and this method for identifying the pycnocline
    depth is less reliable.

    Parameters
    ----------
        nemo_t : xr.Dataset
            nemo object on t-points.
        nemo_w : xr.Dataset, optional
            nemo object on w-points.

    Example basic usage:
    -------------------
        # Create Internal tide diagnostics object
        IT_obj = INTERNALTIDE(nemo_t, nemo_w) # For NEMO objects on t and w-pts
        IT_obj.construct_pycnocline_vars( nemo_t, nemo_w )
        # Make maps of pycnocline thickness and depth
        IT_obj.quick_plot()
    """
    def __init__(self, nemo_t: xr.Dataset, nemo_w: xr.Dataset):  # TODO We're not calling super init...
        debug(f"Creating new {get_slug(self)}")
        self.dataset = xr.Dataset()

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in z_dim, ydim, xdim
        self.nt = nemo_t.dataset.dims['t_dim']
        self.nz = nemo_t.dataset.dims['z_dim']
        self.ny = nemo_t.dataset.dims['y_dim']
        self.nx = nemo_t.dataset.dims['x_dim']
        debug(f"Initialised {get_slug(self)}")

    def construct_pycnocline_vars( self, nemo_t: xr.Dataset, nemo_w: xr.Dataset, strat_thres = -0.01):
        """
        Computes depth moments of stratifcation. Under the assumption that the
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
        nemo_t : xr.Dataset
            nemo object on t-points.
        nemo_w : xr.Dataset, optional
            nemo object on w-points.
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
        fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
        nemo_t = coast.NEMO(dn_files + fn_nemo_grid_t_dat,
                     dn_files + fn_nemo_dom, grid_ref='t-grid')
        # create an empty w-grid object, to store stratification
        nemo_w = coast.NEMO( fn_domain = dn_files + fn_nemo_dom,
                           grid_ref='w-grid')

        # initialise Internal Tide object
        IT = coast.INTERNALTIDE(nemo_t, nemo_w)
        # Construct pycnocline variables: depth and thickness
        IT.construct_pycnocline_vars( nemo_t, nemo_w )
        # Plot pycnocline depth and thickness
        IT.quickplot()

        """
        debug(f"Constructing pycnocline variables for {get_slug(self)}")
        #%% Contruct in-situ density if not already done
        if not hasattr(nemo_t.dataset, 'density'):
            nemo_t.construct_density( EOS='EOS10' )

        #%% Construct stratification if not already done. t-pts --> w-pts
        if not hasattr(nemo_w.dataset, 'rho_dz'):
            nemo_w = nemo_t.differentiate( 'density', dim='z_dim', out_varstr='rho_dz', out_obj=nemo_w ) # --> sci_nwes_w.rho_dz

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in z_dim, ydim, xdim
        nt = nemo_t.dataset.dims['t_dim']
        nz = nemo_t.dataset.dims['z_dim']
        ny = nemo_t.dataset.dims['y_dim']
        nx = nemo_t.dataset.dims['x_dim']

        ## Create a mask for weakly stratified waters
        # Preprocess stratification
        strat = copy.copy(nemo_w.dataset.rho_dz)  # (t_dim, z_dim, ydim, xdim). w-pts.
        # Ensure surface value is 0
        strat[:,0,:,:] = 0
        # Ensure bed value is 0
        strat[:,-1,:,:] = 0
        # mask out the Nan values
        strat = strat.where( ~xr.ufuncs.isnan(nemo_w.dataset.rho_dz), drop=False )
        # create mask with a stratification threshold
        strat_m = nemo_w.dataset.latitude * 0 + 1 # create a stratification mask: [1/0] = strat/un-strat
        strat_m = strat_m.where( strat.min(dim='z_dim').squeeze() < strat_thres, 0, drop=False)
        strat_m = strat_m.transpose("t_dim", "y_dim", "x_dim", transpose_coords=True)

        ## Compute statification variables
        # initialise pycnocline variables
        zd = np.zeros((nt, ny, nx)) # pycnocline depth
        zt = np.zeros((nt, ny, nx)) # pycnocline thickness

        ## Construct intermediate variables
        # Broadcast to fill out missing (time) dimensions in grid data
        _, depth_0_4d = xr.broadcast(strat, nemo_w.dataset.depth_0)
        _, e3_0_4d    = xr.broadcast(strat, nemo_w.dataset.e3_0.squeeze())

        # intergrate strat over depth
        intN2  = ( strat * e3_0_4d ).sum( dim='z_dim', skipna=True)
        # intergrate (depth * strat) over depth
        intzN2 = (strat * e3_0_4d * depth_0_4d).sum( dim='z_dim', skipna=True)

        # compute pycnocline depth
        zd = (intzN2 / intN2 )# pycnocline depth

        # compute pycnocline thickness
        intz2N2 = ( xr.ufuncs.square(depth_0_4d - zd) * e3_0_4d * strat  ).sum( dim='z_dim', skipna=True )
        zt = ( xr.ufuncs.sqrt(intz2N2 / intN2) ) # pycnocline thickness

        # Define xarray attributes
        coords = {'time': (('t_dim'), nemo_t.dataset.time.values),
                    'latitude': (('y_dim','x_dim'), nemo_t.dataset.latitude.values),
                    'longitude': (('y_dim','x_dim'), nemo_t.dataset.longitude.values)}
        dims = ['t_dim', 'y_dim', 'x_dim']

        # Save a xarray objects
        self.dataset['strat_2nd_mom'] = xr.DataArray( zt,
                    coords=coords, dims=dims)
        self.dataset.strat_2nd_mom.attrs['units'] = 'm'
        self.dataset.strat_2nd_mom.attrs['standard_name'] = 'pycnocline thickness'
        self.dataset.strat_2nd_mom.attrs['long_name'] = 'Second depth moment of stratification'


        self.dataset['strat_1st_mom'] = xr.DataArray( zd,
                    coords=coords, dims=dims )
        self.dataset.strat_1st_mom.attrs['units'] = 'm'
        self.dataset.strat_1st_mom.attrs['standard_name'] = 'pycnocline depth'
        self.dataset.strat_1st_mom.attrs['long_name'] = 'First depth moment of stratification'


        #%% Mask pycnocline variables in weak stratification
        #zd_m = zd.where( strat_m > 0, -999, drop=False )
        #zt_m = zt.where( strat_m > 0, -999, drop=False )
        zd_m = zd.where( strat_m > 0 )
        zt_m = zt.where( strat_m > 0 )

        self.dataset['mask'] = xr.DataArray( strat_m,
                    coords=coords, dims=dims)

        self.dataset['strat_2nd_mom_masked'] = xr.DataArray( zt_m,
                    coords=coords, dims=dims)
        self.dataset.strat_2nd_mom_masked.attrs['units'] = 'm'
        self.dataset.strat_2nd_mom_masked.attrs['standard_name'] = 'masked pycnocline thickness'
        self.dataset.strat_2nd_mom_masked.attrs['long_name'] = 'Second depth moment of stratification, masked in weak stratification'


        self.dataset['strat_1st_mom_masked'] = xr.DataArray( zd_m,
                    coords=coords, dims=dims )
        self.dataset.strat_1st_mom_masked.attrs['units'] = 'm'
        self.dataset.strat_1st_mom_masked.attrs['standard_name'] = 'masked pycnocline depth'
        self.dataset.strat_1st_mom_masked.attrs['long_name'] = 'First depth moment of stratification, masked in weak stratification'

    def quick_plot(self, var : xr.DataArray = None):
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
        IT.quick_plot( 'strat_1st_mom_masked' )

        """
        import matplotlib.pyplot as plt

        debug(f"Generating quick plot for {get_slug(self)}")

        if var is None:
            var_lst = [self.dataset.strat_1st_mom_masked, self.dataset.strat_2nd_mom_masked]
        else:
            var_lst = [self.dataset[var]]

        for var in var_lst:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            plt.pcolormesh( self.dataset.longitude.squeeze(),
                           self.dataset.latitude.squeeze(),
                           var.isel(t_dim = 0) )
            #               var.mean(dim = 't_dim') )
            #plt.contourf( self.dataset.longitude.squeeze(),
            #               self.dataset.latitude.squeeze(),
            #               var.mean(dim = 't_dim'), levels=(0,10,20,30,40) )
            title_str = self.dataset.time[0].dt.strftime("%d %b %Y: ").values \
                + var.attrs['standard_name'] +  ' (' + var.attrs['units'] + ')'
            plt.title(title_str)
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.clim([0, 50])
            plt.colorbar()
            plt.show()
        return fig, ax  # TODO if var_lst is empty this will cause an error
