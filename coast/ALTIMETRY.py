from .COAsT import COAsT
from .OBSERVATION import OBSERVATION
from warnings import warn
import numpy as np
import xarray as xa


class ALTIMETRY(OBSERVATION):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def set_dimension_mapping(self):
        self.dim_mapping = None
        
    def set_variable_mapping(self):
        self.var_mapping = None
