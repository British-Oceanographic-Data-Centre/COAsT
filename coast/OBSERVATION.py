from .COAsT import COAsT
import numpy as np
from .logging_util import get_slug, debug, info, warn, warning, error


class OBSERVATION(COAsT):
    def set_dimension_mapping(self):
        self.dim_mapping = None
        debug(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        debug(f"Subsetting {get_slug(self)} indices in {lonbounds}, {latbounds}")
        lon = self.dataset.longitude.copy()
        lat = self.dataset.latitude
        lon[lon>180] = lon[lon>180] - 360
        lon[lon<-180] = lon[lon<-180] + 360
        ff1 = ( lon > lonbounds[0] ).astype(int)  # FIXME This should fail? We can just treat bools as ints here...
        ff2 = ( lon < lonbounds[1] ).astype(int)
        ff3 = ( lat > latbounds[0] ).astype(int)
        ff4 = ( lat < latbounds[1] ).astype(int)
        indices = np.where( ff1 * ff2 * ff3 * ff4 )
        return indices[0]

    def adjust_longitudes(self, lonbounds=[-180,180]):  # TODO This [list] should probably be a (tuple)
        debug(f"Adjusting {get_slug(self)} longitudes with lonbounds {lonbounds}")
        bool0 = self['longitude']<lonbounds[0]
        bool1 = self['longitude']>lonbounds[1]
        self['longitude'][bool0] = self['longitude'][bool0] + 360
        self['longitude'][bool1] = self['longitude'][bool1] - 360
