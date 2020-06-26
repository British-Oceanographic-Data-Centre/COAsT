import numpy as np
import xarray as xr
from warnings import warn
import matplotlib.pyplot as plt
#from .CRPS import CRPS
#from .interpolate_along_dimension import interpolate_along_dimension

class DIAGNOSTICS():
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
    def __init__(self, nemo: xr.Dataset, domain: xr.Dataset):

        self.nemo   = nemo
        self.domain = domain

        # These are bespoke to the internal tide problem
        self.zt = None
        self.zd = None

        # This might be generally useful and could be somewhere more accessible?
        self.strat = None

        self.domain.construct_depths_from_spacings() # compute depths on t and w points
        """
        # These would be generally useful and should be in the NEMO class
        self.depth_t = xr.DataArray( domain.dataset.e3t_0.cumsum( dim='z_dim' ).squeeze() ) # size: nz,my,nx
        self.depth_t.attrs['units'] = 'm'
        self.depth_t.attrs['standard_name'] = 'depth_at_t-points'

        self.depth_w = xr.DataArray( domain.dataset.e3w_0.cumsum( dim='z_dim' ).squeeze() ) # size: nz,my,nx
        self.depth_w.attrs['units'] = 'm'
        self.depth_w.attrs['standard_name'] = 'depth_at_w-points'
        """

        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in z_dim, ydim, xdim
        self.nt = nemo.dataset.dims['t_dim']
        self.nz = nemo.dataset.dims['z_dim']
        self.ny = nemo.dataset.dims['y_dim']
        self.nx = nemo.dataset.dims['x_dim']
        if domain.dataset.dims['z_dim'] != self.nz:
            print('z_dim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['z_dim'], self.nz))
        if domain.dataset.dims['y_dim'] != self.ny:
            print('ydim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['y_dim'], self.ny))
        if domain.dataset.dims['x_dim'] != self.nx:
            print('xdim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['x_dim'], self.nx))


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
        self.strat = self.difftpt2tpt( var, dim='z_dim' ) \
                    / self.difftpt2tpt( self.domain.depth_t, dim='z_dim' )

        # Add attributes
        if 'standard_name' not in var.attrs.keys(): var.attrs['standard_name'] = '[var]'
        if 'units' not in var.attrs.keys(): var.attrs['units'] = '[var]'
        self.strat.attrs['units'] = var.units + '/m'
        self.strat.attrs['standard_name'] = var.standard_name + ' stratification'



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
        self.get_stratification( self.nemo.dataset.votemper )

        print('Using only temperature for stratification at the moment')
        N2_4d = self.strat  # (t_dim, z_dim, ydim, xdim). T-pts. Surface value == 0

        # Ensure surface value is 0
        N2_4d[:,0,:,:] = 0
        # Ensure bed value is 0
        N2_4d[:,-1,:,:] = 0

        # mask out the Nan values
        N2_4d = N2_4d.where( xr.ufuncs.isnan(self.nemo.dataset.votemper), drop=True )
        #N2_4d[ np.where( np.isnan(self.nemo.dataset.votemper) ) ] = np.NaN

        # initialise variables
        z_d = np.zeros((self.nt, self.ny, self.nx)) # pycnocline depth
        z_t = np.zeros((self.nt, self.ny, self.nx)) # pycnocline thickness


        # Broadcast to fill out missing (time) dimensions in grid data
        _, depth_t_4d = xr.broadcast(N2_4d, self.domain.depth_t)
        _, depth_w_4d = xr.broadcast(N2_4d, self.domain.depth_w)
        _, e3t_0_4d   = xr.broadcast(N2_4d, self.domain.dataset.e3t_0.squeeze())


        # intergrate strat over depth
        intN2  = ( N2_4d * e3t_0_4d ).sum( dim='z_dim', skipna=True)
        # intergrate (depth * strat) over depth
        intzN2 = (N2_4d * e3t_0_4d * depth_t_4d).sum( dim='z_dim', skipna=True)


        # compute pycnocline depth
        z_d = intzN2 / intN2 # pycnocline depth

        # compute pycnocline thickness
        intz2N2 = ( xr.ufuncs.square(depth_t_4d - z_d) * e3t_0_4d * N2_4d  ).sum( dim='z_dim', skipna=True )
        #    intz2N2 = np.trapz( (z-z_d_tile)**2 * N2, z, axis=ax)
        z_t = xr.ufuncs.sqrt(intz2N2 / intN2)


        self.zt = xr.DataArray( z_t )
        self.zt.attrs['units'] = 'm'
        self.zt.attrs['standard_name'] = 'pycnocline thickness'

        self.zd = xr.DataArray( z_d )
        self.zd.attrs['units'] = 'm'
        self.zd.attrs['standard_name'] = 'pycnocline depth'
