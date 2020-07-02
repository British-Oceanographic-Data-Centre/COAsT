import numpy as np
import xarray as xr
from warnings import warn
import copy
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
    WIP: IT_obj.get_stratification( sci_nwes.dataset.votemper ) # --> self.strat
    WIP: IT_obj.get_pyc_vars()
    WIP: IT_obj.quick_plot()
    '''
    def __init__(self, nemo: xr.Dataset, domain: xr.Dataset):

        self.nemo   = nemo
        self.domain = domain

        # These are bespoke to the internal tide problem
        self.zt = None
        self.zd = None

        # This might be generally useful and could be somewhere more accessible?
        self.strat = None

        """
        ## Jeff's original hack job for depth variables
        #self.domain.construct_depths_from_spacings() # compute depths on t and w points
        # Anthony's original numpy method for depth variables
        self.domain.depth_t, self.domain.depth_w = self.domain.get_depth(
                                                    self.domain.dataset.e3t_0.values,
                                                    self.domain.dataset.e3w_0.values)
        """
        # xarray method for parsing depth variables:  # --> domain.depth_t , domain.depth_w
        self.domain.get_depth_as_xr(self.domain.dataset.e3t_0,
                                    self.domain.dataset.e3w_0)

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


    def diff_w_r_t(self, var : xr.DataArray, dim='z_dim'):
        """
        Differentiate input var with respect to the grid it is on over the
        given dimension

        return new variable with appropriate attributes:
            units
            grid
            standard_name

        self includes the domain information
        var has an attribute stating the grid.
        """
        var_derivative = None
        new_grid = ""
        new_standard_name = ""
        new_units = ""

        nt = var.sizes['t_dim']
        nz = var.sizes['z_dim']
        ny = var.sizes['y_dim']
        nx = var.sizes['x_dim']

        if (var.attrs['grid'] == 't-grid') and (dim == 'z_dim'):
            new_grid = 'w-grid'
            # create new DataArray with the same dimensions as the parent
            # Crucially have a coordinate value that is appropriate to the target location.
            blank = xr.DataArray(np.zeros((nt,1,ny,nx)),
                        coords={ 'deptht': ('z_dim',[0])},
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
