from .COAsT import COAsT
import numpy as np
from .logging_util import get_slug, debug, info, warn, warning, error


class OBSERVATION(COAsT):
    def set_dimension_mapping(self):
        self.dim_mapping = None
        debug(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")

    def adjust_longitudes(self, lonbounds=[-180,180]):  # TODO This [list] should probably be a (tuple)
        debug(f"Adjusting {get_slug(self)} longitudes with lonbounds {lonbounds}")
        bool0 = self['longitude']<lonbounds[0]
        bool1 = self['longitude']>lonbounds[1]
        self['longitude'][bool0] = self['longitude'][bool0] + 360
        self['longitude'][bool1] = self['longitude'][bool1] - 360
