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
    for calculation of dynamical diagnostics.
    '''
    def __init__(self, nemo: xr.Dataset):

        self.nemo   = nemo
        self.dataset = nemo.dataset


    def get_density(self, T: xr.DataArray, S: xr.DataArray, z: xr.DataArray):
        """ Compute a density from temperature, salinity """
        self.dataset['rho'] = xr.DataArray( gsw.rho(S,T,z), dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'] )


    def differentiate(self, var : xr.DataArray, dim='z_dim'):
        """
        Differentiate input var with respect to the grid it is on over the
        given dimension

        return new variable with appropriate attributes:
            units
            grid
            standard_name

        self includes the domain information
        var has an attribute stating the grid.


        The derivative is computed as a 1st-order centred difference following the NEMO
         manual.

        The returned object has dimensions and coordinates names that match the parent.

        The returned object has a undated attributes for its:
         grid, name, units.

        The dimension sizes are the same as the parent, with a zero at the end value.

        For terrain following coordinates the depth coordinate is a vector of
         ordered depths. In this case the d/dz inherits depth coordinates are the same
          as the parent, except the surface value, which is zero.

        """
        var_derivative = None
        new_grid = ""
        new_standard_name = ""
        new_units = ""

        nt = var.sizes['t_dim']
        nz = var.sizes['z_dim']
        ny = var.sizes['y_dim']
        nx = var.sizes['x_dim']

        if (var.attrs['grid_ref'] == 't-grid') and (dim == 'z_dim'):
            new_grid = 'w-grid'
            # create new DataArray with the same dimensions as the parent
            # Crucially have a coordinate value that is appropriate to the target location.
            blank = xr.DataArray(np.zeros((nt,1,ny,nx)),
                        coords={ 'depth_0': ('z_dim',[0])},
                        dims=var.dims)
            # Add blank slice to the 'surface'. Concat over the 'dim' coords
            diff = xr.concat([blank, var.diff(dim)], dim)
            diff_ndim, e3w_ndim = xr.broadcast( diff, self.domain.dataset.e3w_0.squeeze() )
            # Finally compute the derivative
            var_derivative = - diff_ndim / e3w_ndim

        else:
            print('Not expecting that possibility')
            pass

        # Define new attributes
        new_standard_name = 'd('+var._name+')/d('+dim+')'
        new_units = var.attrs['units']+'/'+'['+dim+' coord units]'
        # Convert to a xr.DataArray and return
        return xr.DataArray( var_derivative,
                            dims=var.dims,
                            attrs={'grid' : new_grid,
                                   'units': new_units,
                                   'standard_name': new_standard_name})
