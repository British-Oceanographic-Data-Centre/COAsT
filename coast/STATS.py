import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension
from .CRPS import CRPS

class STATS():
    
    def __init__(self, mod_data, mod_dom, obs_object):
        self.mod_data = mod_data
        self.mod_dom = mod_dom
        self.obs_object = obs_object
        return
        
    def __setitem__(self):
        return
        
    def __str__(self):
        return
    
    def crps(self, mod_var:str, obs_var:str, nh_radius: float=111, 
             nh_type: str="radius", cdf_type: str="empirical", 
             time_interp:str="nearest"):
        return CRPS(self.mod_data, self.mod_dom, self.obs_object,
                    mod_var, obs_var, nh_radius, nh_type, cdf_type,
                    time_interp)
    
#    def errors(self, mod_var, obs_var, space_interp: str='nearest',
#               time_interp: str='nearest'):
#        return ERRORS(self.mod_data, self.mod_dom, self.obs_object,
#                    mod_var, obs_var, space_interp, time_interp)
        