from .COAsT import COAsT
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def set_dimension_mapping(self):
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim',
                            'y':'y_dim', 'x':'x_dim'}
        #self.dim_mapping = None

    def set_variable_mapping(self):
        #self.var_mapping = {'time_counter':'time',
        #                    'votemper' : 'temperature',
        #                    'temp' : 'temperature'}
        self.var_mapping = None

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller
