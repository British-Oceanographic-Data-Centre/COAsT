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
        self.var_list = []

    def set_command_variables(self):
        """
         A method to make accessing the following simpler
        """
        try:
            self.sla_filtered = self.dataset.sla_filtered
            self.var_list.append('sla_filtered')
        except AttributeError as e:
            warn(str(e))

        try:
            self.sla_unfiltered = self.dataset.sla_unfiltered
            self.var_list.append('sla_unfiltered')
        except AttributeError as e:
            warn(str(e))

        try:
            self.longitude = self.dataset.longitude
            self.var_list.append('longitude')
            self.adjust_longitudes(self.longitude)
        except AttributeError as e:
            warn(str(e))

        try:
            self.latitude = self.dataset.latitude
            self.var_list.append('latitude')
        except AttributeError as e:
            warn(str(e))
            
        try:
            self.mdt = self.dataset.mdt
            self.var_list.append('mdt')
        except AttributeError as e:
            warn(str(e))

        try:
            self.ocean_tide = self.dataset.ocean_tide
            self.var_list.append('ocean_tide')
        except AttributeError as e:
            warn(str(e))
            
        try:
            self.time = self.dataset.time
            self.var_list.append('time')
        except AttributeError as e:
            warn(str(e))
