from .COAsT import COAsT
from warnings import warn
import numpy as np
import xarray as xr

class OBSERVATION(COAsT):

    def set_dimension_mapping(self):
        self.dim_mapping = None

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        lon = self.dataset.longitude.copy()
        lat = self.dataset.latitude
        lon[lon>180] = lon[lon>180] - 360
        lon[lon<-180] = lon[lon<-180] + 360
        ff1 = ( lon > lonbounds[0] ).astype(int)
        ff2 = ( lon < lonbounds[1] ).astype(int)
        ff3 = ( lat > latbounds[0] ).astype(int)
        ff4 = ( lat < latbounds[1] ).astype(int)
        indices = np.where( ff1 * ff2 * ff3 * ff4 )
        return indices[0]
    
    def adjust_longitudes(self, lonbounds=[-180,180]):
        bool0 = self['longitude']<lonbounds[0]
        bool1 = self['longitude']>lonbounds[1]
        self['longitude'][bool0] = self['longitude'][bool0] + 360
        self['longitude'][bool1] = self['longitude'][bool1] - 360
        
    def interpolate_model_to_obs(self):
        print('Method not implemented for observation object type.')
        
    def quick_plot(self):
        print('Method not implemented for observation object type.')
