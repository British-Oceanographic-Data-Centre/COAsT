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

    def set_command_variables(self):
        """
         A method to make accessing the following simpler
        """
        
        for key, value in self.var_dict.items():
            try:
                setattr( self, key, self.dataset[value] )
                self.var_list.append(key)
            except AttributeError as e:
                warn(str(e))
                
        self.adjust_longitudes()
