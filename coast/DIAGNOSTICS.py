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
