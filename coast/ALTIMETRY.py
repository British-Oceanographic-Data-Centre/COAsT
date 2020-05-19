from .COAsT import COAsT
from warnings import warn
import numpy as np
import xarray as xa


class ALTIMETRY(COAsT):

    def __init__(self):
        super()
        self.path_to_file = None
        self.latitude = None
        self.longitude = None
        self.sossheig = None

    def set_command_variables(self):
        """
         A method to make accessing the following simpler
                bathy_metry (t,y,x) - float - (m i.e. metres)
                nav_lat (y,x) - float - (deg)
                nav_lon (y,x) - float - (deg)
                e1u, e1v, e1t, e1f (t,y,x) - double - (m)
                e2u, e2v, e2t, e2f (t,y,x) - double - (m)
        """
        try:
            self.latitude = self.dataset.latitude
        except AttributeError as e:
            warn(str(e))

        try:
            self.longitude = self.dataset.longitude
        except AttributeError as e:
            warn(str(e))

        try:
            self.sossheig = self.dataset.sossheig
        except AttributeError as e:
            warn(str(e))



   