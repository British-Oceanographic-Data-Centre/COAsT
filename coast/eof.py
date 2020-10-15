import matplotlib.pyplot as plt 
import xarray as xr
import xarray.ufuncs as uf
import numpy as np
from scipy import linalg

class EOF:
    def __init__(self, variable:xr.DataArray, full_matrices=False, time_dim_name:str='t_dim'):
        self.variable = variable.transpose(...,time_dim_name)
        self.__compute(full_matrices)
        
    def __compute(self, full_matrices):
        I,J,T = np.shape(self.variable.data)
        F = np.nan_to_num( np.reshape(self.variable.data,(I*J, T)) )
                      
        # Remove constant zero point such as land points
        active_ind = np.where(F.any(axis=1))[0] 
        A = F[active_ind,:] 
        # Remove time mean at each grid point
        mean = A.mean(axis=1)
        A = A - mean[:, np.newaxis]
        
        # Calculate eofs and pcs using SVD        
        P, D, Q = linalg.svd( A, full_matrices=full_matrices )
        EOFs = np.zeros_like(F, dtype=P.dtype)
        # EOFs if we didn't normalise by std
        EOFs[active_ind,:] = P 
        #EOFs[active_ind,:] = std[:,np.newaxis] * P # if normalised by std, get back units
        
        # Calculate variance explained
        if full_matrices:        
            variance_explained = 100.*( D**2 / np.sum( D**2 ) )
        else:
            Inv = P.dot( np.dot( np.diag(D), Q ) ) # PDQ
            var1 = np.sum( np.var( Inv, axis=1, ddof=1 ) )
            var2 = np.sum( np.var( A, axis=1, ddof=1 ) )
            mult = var1 / var2
            variance_explained = 100.*mult*( D**2 / np.sum( D**2 ) )
        
        # Reshape and scale PCs
        PCs = np.transpose(Q) * D
        scale = np.max(np.abs(PCs),axis=0)
        EOFs = np.reshape(EOFs, (I,J,T)) * scale
        PCs = np.transpose(Q) * D / scale
        
        
        # Assign to xarray variables
        # copy the coordinates 
        coords = {'mode':(('mode'),np.arange(1,T+1))}
        time_coords = {'mode':(('mode'),np.arange(1,T+1))}
        for coord in self.variable.coords:
            if self.variable.dims[2] not in self.variable[coord].dims:
                coords[coord] = (self.variable[coord].dims, self.variable[coord])
            else:
                if self.variable.dims[2] == self.variable[coord].dims:
                    time_coords[coord] = (self.variable[coord].dims, self.variable[coord])
        self.dataset = xr.Dataset()
        dims = (self.variable.dims[:2]) + ('mode',)
        self.dataset['EOF'] = xr.DataArray(EOFs, coords=coords, dims=dims)
        self.dataset.EOF.attrs['standard name'] = 'EOFs'
        self.dataset.EOF.attrs['units'] = self.variable.units
        
        dims = (self.variable.dims[2],'mode')
        self.dataset['PC'] = xr.DataArray(PCs, coords=time_coords, dims=dims)
        
        self.dataset['variance'] = (xr.DataArray(variance_explained, 
                coords={'mode':(('mode'),np.arange(1,T+1))}, dims=['mode']))
                
        
        