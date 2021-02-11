from .COAsT import COAsT
import xarray as xr
import numpy as np
import skimage.draw as draw
from . import general_utils

class MASK_MAKER(): 

    def __init__(self):
        
        return
    
    @staticmethod
    def fill_polygon_by_index(array_to_fill, vertices_r, vertices_c, 
                              fill_value = 1, additive = False):
        """
        Draws and fills a polygon onto an existing numpy array based on array
        indices. To create a new mask, give np.zeros(shape) as input. 
        Polygon vertices are drawn in the order given.
        
        Parameters
        ----------
        array_to_fill (2D array): Array onto which to fill polygon
        vertices_r (1D array): Row indices for polygon vertices
        vertices_c (1D_array): Column indices for polygon vertices
        fill_value (float, bool or int): Fill value for polygon (Default: 1)
        additive (bool): If true, add fill value to existing array. Otherwise
                         indices will be overwritten. (Default: False)

        Returns
        -------
        Filled 2D array
        """
        array_to_fill = np.array(array_to_fill)
        polygon_ind = draw.polygon(vertices_r, vertices_c, 
                                       array_to_fill.shape)
        if additive:
            array_to_fill[polygon_ind[0], polygon_ind[1]] += fill_value
        else:
            array_to_fill[polygon_ind[0], polygon_ind[1]] = fill_value
        return array_to_fill
    
    @staticmethod
    def fill_polygon_by_lonlat(array_to_fill, lon_array, lat_array, 
                               vertices_lon, vertices_lat, fill_value = 1,
                               additive = False):
        """
        Draws and fills a polygon onto an existing numpy array based on 
        vertices defined by longitude and latitude locations. This does NOT
        draw a polygon on a sphere, but instead based on straight lines 
        between points. This is OK for small regional areas, but not advisable
        for large and global regions.
        Polygon vertices are drawn in the order given.
        
        Parameters
        ----------
        array_to_fill (2D array): Array onto which to fill polygon
        vertices_r (1D array): Row indices for polygon vertices
        vertices_c (1D_array): Column indices for polygon vertices
        fill_value (float, bool or int): Fill value for polygon (Default: 1)
        additive (bool): If true, add fill value to existing array. Otherwise
                         indices will be overwritten. (Default: False)

        Returns
        -------
        Filled 2D array
        """
        array_to_fill = np.array(array_to_fill)
        ind2D = general_utils.nearest_indices_2D(lon_array, lat_array, 
                                                 vertices_lon, vertices_lat)
        
        polygon_ind = draw.polygon(ind2D[0], ind2D[1], array_to_fill.shape)
        if additive:
            array_to_fill[polygon_ind[0], polygon_ind[1]] += fill_value
        else:
            array_to_fill[polygon_ind[0], polygon_ind[1]] = fill_value
        return array_to_fill