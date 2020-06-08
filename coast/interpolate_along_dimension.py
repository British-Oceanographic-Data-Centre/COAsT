import xarray as xr
import numpy as np
from .COAsT import COAsT

class interpolate_along_dimension():
    '''
    An object for flexibly and quickly interpolating an xarray variable along
    a specified dimension.
    '''
    
    def __init__(self, data, dim_interp, dim_name, method='nearest'):
        self.method = method
        self.dim_name = dim_name
        self.data = data
        self.weights = self.calculate_weights(self.data[dim_name], 
                                              dim_interp)
        
    def __getitem__(self, indices):
        if self.method == 'nearest':
            return self.get_nearest(self.data, indices)
        else:
            raise NotImplementedError
    
    def get_nearest(self, data, indices):
        ''' Returns the nearest interpolated data for specified indices '''
        return data.isel({self.dim_name : self.weights[indices]})
    
    def get_linear(self):
        ''' Returns the linearly interpolated data for specified indices '''
        raise NotImplementedError
        return
    
    def calculate_weights(self, x0, xi):
        ''' Control routine for sending data to different weights functions '''
        if self.method == 'nearest':
            weights = self.calculate_weights_nearest(x0, xi)
        else:
            raise NotImplementedError
        return weights
        
    def calculate_weights_nearest(self, x0, xi):
        ''' Calculates weights for nearest neighbour. In this case weights are
        just the indices of the data that is closest. '''
        vdiff = self.difference_matrix(x0, xi)
        return np.argmin(vdiff, axis=0)
        
    def calculate_weights_linear(self, x0, xi):
        ''' Calculates weights for linear interpolation. The form of each row 
        of the weights array is [w1, w2, i1, i2]. w1 and w2 are the weights and
        i1, i2 are the indices of the data that the weights are applied to'''
        raise NotImplementedError
        return

    def difference_matrix(self, v1, v2):
        ''' Calculates all pairwise absolute distances between two vectors.'''
        mg_x, mg_y = np.meshgrid(v1,v2)
        return np.transpose( np.abs(mg_x - mg_y) )