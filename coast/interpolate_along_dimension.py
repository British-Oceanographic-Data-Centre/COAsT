import xarray as xr
import numpy as np
from .COAsT import COAsT

class interpolate_along_dimension():
    
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
        return data[self.weights[indices], :, :]
    
    def get_linear(self):
        raise NotImplementedError
        return
    
    def calculate_weights(self, x0, xi):
        if self.method == 'nearest':
            weights = self.calculate_weights_nearest(x0, xi)
        else:
            raise NotImplementedError
        return weights
        
    def calculate_weights_nearest(self, x0, xi):
        vdiff = self.difference_matrix(x0, xi)
        return np.argmin(vdiff, axis=0)
        
    def calculate_weights_linear(self):
        raise NotImplementedError
        return

    def difference_matrix(self, v1, v2):
        mg_x, mg_y = np.meshgrid(v1,v2)
        return np.transpose( np.abs(mg_x - mg_y) )
    
    def find_dimension(self, data, dim_name):
        count = 0
        for item in data.dims:
            if item == dim_name: return count 
            else: count = count+1
        raise Exception('Dimension not found in DataArray')