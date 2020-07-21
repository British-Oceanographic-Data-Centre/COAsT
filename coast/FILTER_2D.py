import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy
from .NEMO import NEMO
from .COAsT import COAsT
import scipy as sp

class FILTER_2D():
    '''
    
    '''
    
    def __init__(self, model, var_name, filter_type: str='gaussian'):
        self.dataset = model.dataset[var_name]
        
        dims = self.dataset.dims
        if 'z_dim' in dims:
            self.dataset = self.dataset.isel(z_dim=0)
        
        if 't_dim' in dims:
            self.dataset = self.dataset.isel(t_dim=0)
        
        self.filtered = self.gaussian_filter(self.dataset, 5, truncate=10)

        
    def gaussian_filter(self, array, sigma, **kwargs):
        
        A = np.array(array)
            
        V=A.copy()
        V[np.isnan(A)]=0
        VV=sp.ndimage.gaussian_filter(V,sigma=sigma,**kwargs)

        W=0*A.copy()+1
        W[np.isnan(A)]=0
        WW=sp.ndimage.gaussian_filter(W,sigma=sigma,**kwargs)
        
        WW = VV/WW
        WW[np.isnan(A)] = np.nan
        
        return WW
        