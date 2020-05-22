from .COAsT import COAsT
from warnings import warn
import numpy as np
import xarray as xa


class OBSERVATION(COAsT):

    def __init__(self):
        super()
        
    def extract_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        lon = self.longitude.copy()
        lat = self.latitude
        lon[lon>180] = lon[lon>180] - 360
        lon[lon<-180] = lon[lon<-180] + 360
        ff1 = ( lon > lonbounds[0] ).astype(int)
        ff2 = ( lon < lonbounds[1] ).astype(int)
        ff3 = ( lat > latbounds[0] ).astype(int)
        ff4 = ( lat < latbounds[1] ).astype(int)
        indices = np.where( ff1 * ff2 * ff3 * ff4 )
        self.extract_indices_all_var(indices)
        return indices
    
    def extract_indices_all_var(self, indices):
        """ 
        Extracts indices from all variables and saves them into obs object.
        """
        for var in self.var_list:
            setattr(self, var, getattr(self, var)[indices])
        return
    
    def adjust_longitudes(self, lonbounds=[-180,180]):
        bool0 = self.longitude<lonbounds[0]
        bool1 = self.longitude>lonbounds[1]
        self.longitude[bool0] = self.longitude[bool0] + 360
        self.longitude[bool1] = self.longitude[bool1] - 360
