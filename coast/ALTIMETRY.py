from .COAsT import COAsT
from .OBSERVATION import OBSERVATION
from warnings import warn
import numpy as np
import xarray as xa


class ALTIMETRY(OBSERVATION):

    def __init__(self):
        super()
        self.sla_filtered = None
        self.sla_unfiltered = None
        self.mdt = None
        self.ocean_tide = None
        self.latitude = None
        self.longitude = None
        self.time = None
        # List of variables that are actually in the object (successfully read)
        self.var_list = []
        # Mapping of quick access variables to dataset variables
        # {'referencing_var' : 'dataset_var'}.
        self.var_dict = {'sla_filtered'   : 'sla_filtered',
                         'sla_unfiltered' : 'sla_unfiltered',
                         'mdt'            : 'mdt',
                         'ocean_tide'     : 'ocean_tide',
                         'longitude'      : 'longitude', 
                         'latitude'       : 'latitude',
                         'time'           : 'time'}

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
