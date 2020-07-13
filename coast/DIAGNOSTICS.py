from .COAsT import COAsT
import numpy as np
import xarray as xr
from warnings import warn
import copy
import gsw
#from .CRPS import CRPS
#from .interpolate_along_dimension import interpolate_along_dimension

class DIAGNOSTICS(COAsT):
    '''
    Object for handling and storing necessary information, methods and outputs
    for calculation of dynamical diagnostics. The object is
    initialized by passing it COAsT variables of model data, model domain.

    Example basic usage::

    # Create Diagnostics object
    IT_obj = Diagnostics(sci_nwes, dom_nwes)
    # Construct stratification
    IT_obj.get_stratification( sci_nwes.dataset.votemper ) # --> self.strat

    IT_obj.get_pyc_vars()

    import matplotlib.pyplot as plt

    plt.pcolor( IT_obj.strat[0,10,:,:]); plt.show()

    plt.plot( IT_obj.strat[0,:,100,60],'+'); plt.show()

    plt.plot(sci_nwes.dataset.votemper[0,:,100,60],'+'); plt.show()
    '''
    def __init__(self, nemo: xr.Dataset):

        self.nemo   = nemo

        # These are bespoke to the internal tide problem
        self.dataset = nemo.dataset
        #self.zt = None
        #self.zd = None

        # This might be generally useful and could be somewhere more accessible?
        #self.strat = None

        #self.domain.construct_depths_from_spacings() # compute depths on t and w points
        self.depth_0 = self.dataset.depth_0

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in z_dim, ydim, xdim
        self.nt = nemo.dataset.dims['t_dim']
        self.nz = nemo.dataset.dims['z_dim']
        self.ny = nemo.dataset.dims['y_dim']
        self.nx = nemo.dataset.dims['x_dim']
        #if domain.dataset.dims['z_dim'] != self.nz:
        #    print('z_dim domain data size (%s) differs from nemo data size (%s)'
        #          %(domain.dataset.dims['z_dim'], self.nz))
        #if domain.dataset.dims['y_dim'] != self.ny:
        #    print('ydim domain data size (%s) differs from nemo data size (%s)'
        #          %(domain.dataset.dims['y_dim'], self.ny))
        #if domain.dataset.dims['x_dim'] != self.nx:
        #    print('xdim domain data size (%s) differs from nemo data size (%s)'
        #          %(domain.dataset.dims['x_dim'], self.nx))


    def difftpt2tpt(self, var, dim='z_dim'):
        """
        Compute the Euler derivative of T-pt variable onto a T-pt.
        Input the dimension index for derivative
        """
        if dim == 'z_dim':
            difference = 0.5*( var.roll(z_dim=-1, roll_coords=True)
                    - var.roll(z_dim=+1, roll_coords=True) )
        else:
            print('Not expecting that dimension yet')
        return difference
    

    def get_stratification(self, var: xr.DataArray ):
        """
        Compute centered vertical difference on T-points
        """
        self.dataset['strat'] = self.difftpt2tpt( var, dim='z_dim' ) \
                    / self.difftpt2tpt( self.dataset.depth_0, dim='z_dim' )

        # Add attributes
        if 'standard_name' not in var.attrs.keys(): var.attrs['standard_name'] = '[var]'
        if 'units' not in var.attrs.keys(): var.attrs['units'] = '[var]'
        self.dataset.strat.attrs['units'] = var.units + '/m'
        self.dataset.strat.attrs['standard_name'] = var.standard_name + ' stratification'

    def get_deriv(self, var: xr.DataArray ):
        """
        Compute centered vertical difference on T-points
        """
        return self.difftpt2tpt( var, dim='z_dim' ) \
                    / self.difftpt2tpt( self.dataset.depth_0, dim='z_dim' )
                    

    def get_density(self, T: xr.DataArray, S: xr.DataArray, z: xr.DataArray):
        """ Compute a density from temperature, salinity """
        self.dataset['rho'] = xr.DataArray( gsw.rho(S,T,z), dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'] )
        

    def get_pyc_vars(self):
        """

        Pycnocline depth: z_d = \int zN2 dz / \int N2 dz
        Pycnocline thickness: z_t = \sqrt{\int (z-z_d)^2 N2 dz / \int N2 dz}

        Computes stratification on T-points
        Computes pycnocline variables with T-points depths and thicknesses

        Output:
            self.zd - (t,y,x) pycnocline depth
            self.zt - (t,y,x) pycnocline thickness
        Useage:
            ...
        """

        # compute stratification
        #self.get_stratification( self.nemo.dataset.votemper )
        #self.get_stratification( self.nemo.dataset.thetao )
        #print('Using only temperature for stratification at the moment')

        #self.get_stratification( self.dataset.rho  ) # --> self.strat

        N2_4d = copy.copy(self.dataset.strat)  # (t_dim, z_dim, ydim, xdim). T-pts. Surface value == 0

        # Ensure surface value is 0
        N2_4d[:,0,:,:] = 0

        # Ensure bed value is 0
        N2_4d[:,-1,:,:] = 0


        # mask out the Nan values
        N2_4d = N2_4d.where( ~xr.ufuncs.isnan(self.nemo.dataset.temperature), drop=True )
        #N2_4d[ np.where( np.isnan(self.nemo.dataset.votemper) ) ] = np.NaN

        # initialise variables
        z_d = np.zeros((self.nt, self.ny, self.nx)) # pycnocline depth
        z_t = np.zeros((self.nt, self.ny, self.nx)) # pycnocline thickness


        # Broadcast to fill out missing (time) dimensions in grid data
        _, depth_0_4d = xr.broadcast(self.dataset.strat, self.dataset.depth_0)
        _, e3_0_4d    = xr.broadcast(self.dataset.strat, self.dataset.e3_0.squeeze())

        
        # intergrate strat over depth
        intN2  = ( N2_4d * e3_0_4d ).sum( dim='z_dim', skipna=True)
        # intergrate (depth * strat) over depth
        intzN2 = (N2_4d * e3_0_4d * depth_0_4d).sum( dim='z_dim', skipna=True)


        # compute pycnocline depth
        z_d = intzN2 / intN2 # pycnocline depth

        # compute pycnocline thickness
        intz2N2 = ( xr.ufuncs.square(depth_0_4d - z_d) * e3_0_4d * N2_4d  ).sum( dim='z_dim', skipna=True )
        #    intz2N2 = np.trapz( (z-z_d_tile)**2 * N2, z, axis=ax)
        z_t = xr.ufuncs.sqrt(intz2N2 / intN2)


        self.dataset['zt'] = xr.DataArray( z_t )
        self.dataset.zt.attrs['units'] = 'm'
        self.dataset.zt.attrs['standard_name'] = 'pycnocline thickness'

        self.dataset['zd'] = xr.DataArray( z_d )
        self.dataset.zd.attrs['units'] = 'm'
        self.dataset.zd.attrs['standard_name'] = 'pycnocline depth'
        
    def quick_plot(self, var : xr.DataArray = None):
        #try:
        #    import cartopy.crs as ccrs  # mapping plots
        #    import cartopy.feature  # add rivers, regional boundaries etc
        #    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
        #    from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        #except ImportError:
        #    import sys
        #    warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
        #    sys.exit(-1)
            
        import matplotlib.pyplot as plt
        #fig = plt.figure(figsize=(10, 10))
        #ax = fig.gca()
        #ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        if var is None:
            var_lst = [self.zt, self.zd]
        else: 
            var_lst = var
            
        for var in var_lst:
            plt.figure(figsize=(10, 10))
            plt.pcolormesh( self.dataset.latitude.squeeze(), 
                           self.dataset.longitude.squeeze(),
                           var.mean(dim = 't_dim') )
            plt.title(var.attrs['standard_name'] +  ' (' + var.attrs['units'] + ')')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.clim([0, 50])
            plt.colorbar()
            plt.show()
