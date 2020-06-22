import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt
from .CDF import CDF
from .interpolate_along_dimension import interpolate_along_dimension

class ERRORS():
    
    def __init__(self, mod_data, mod_dom, obs_object, 
                 mod_var:str, obs_var:str):
        self.mod_data = mod_data
        self.mod_dom = mod_dom
        self.obs_object = obs_object
        self.mod_var = mod_var
        self.obs_var = obs_var
        self.longitude   = obs_object['longitude']
        self.latitude    = obs_object['latitude']
        self.err = None
        self.abs_err = None
        self.corr = None
        self.mean_err = None
        self.mae = None
        self.rmse = None
        self.calculate()
        return
        
    def calculate():
        
        
        
        return