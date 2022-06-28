"""Mask maker"""
import xarray as xr
import numpy as np
import skimage.draw as draw
from . import general_utils


class MaskMaker:
    def __init__(self):

        return

    @staticmethod
    def make_mask_dataset(longitude, latitude, mask_list):
        if type(mask_list) is not list:
            mask_list = [mask_list]
        gridded_mask = xr.Dataset()
        gridded_mask["longitude"] = (["y_dim", "x_dim"], longitude)
        gridded_mask["latitude"] = (["y_dim", "x_dim"], latitude)
        n_masks = len(mask_list)
        nr, nc = mask_list[0].shape
        all_masks = np.zeros((n_masks, nr, nc))
        gridded_mask["mask"] = (["dim_mask", "y_dim", "x_dim"], all_masks)
        for mm in np.arange(n_masks):
            gridded_mask["mask"][mm] = mask_list[mm]
        return gridded_mask

    @staticmethod
    def fill_polygon_by_index(array_to_fill, vertices_r, vertices_c, fill_value=1, additive=False):
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
        polygon_ind = draw.polygon(vertices_r, vertices_c, array_to_fill.shape)
        if additive:
            array_to_fill[polygon_ind[0], polygon_ind[1]] += fill_value
        else:
            array_to_fill[polygon_ind[0], polygon_ind[1]] = fill_value
        return array_to_fill

    @staticmethod
    def fill_polygon_by_lonlat(
        array_to_fill, longitude, latitude, vertices_lon, vertices_lat, fill_value=1, additive=False
    ):
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
        ind_2d = general_utils.nearest_indices_2d(longitude, latitude, vertices_lon, vertices_lat)

        polygon_ind = draw.polygon(ind_2d[1], ind_2d[0], array_to_fill.shape)
        if additive:
            array_to_fill[polygon_ind[0], polygon_ind[1]] += fill_value
        else:
            array_to_fill[polygon_ind[0], polygon_ind[1]] = fill_value
        return array_to_fill

    @classmethod
    def region_def_nws_north_sea(cls, longitude, latitude, bath):
        """
        Regional definition for the North Sea (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [-5.34, -0.7, 7.5, 7.5, 9, 9, 6.3, 6.3, 5, 5, 4.126, 4.126, -1.071]
        vertices_lat = [56.93, 54.09, 54.09, 56, 56, 57.859, 57.859, 58.121, 58.121, 58.59, 58.59, 60.5, 60.5]

        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_outer_shelf(cls, longitude, latitude, bath):
        """
        Regional definition for the Outher Shelf (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [-4, -9.5, -1, 3.171, 3.171, -3.76, -3.76, -12, -12, -12, -4]
        vertices_lat = [50.5, 52.71, 60.5, 60.45, 63.3, 63.3, 60.45, 60.45, 55.28, 48, 48]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_norwegian_trench(cls, longitude, latitude, bath):
        """
        Regional definition for the Norwegian Trench (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [10.65, 1.12, 1.12, 10.65]
        vertices_lat = [61.83, 61.83, 48, 48]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath > 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_english_channel(cls, longitude, latitude, bath):
        """
        Regional definition for the English Channel (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [7.57, 7.57, -0.67, -2, -3.99, -3.99, -3.5, 12, 14]
        vertices_lat = [56, 54.08, 54.08, 50.7, 50.7, 48.8, 48, 48, 56]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_south_north_sea(cls, longitude, latitude, bath):
        vertices_lon = [-0.67, -0.67, 9, 9, 7.57, 7.57]
        vertices_lat = [54.08, 51, 51, 56, 56, 54.08]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~xr.ufuncs.isnan(bath))
        return mask

    @classmethod
    def region_def_off_shelf(cls, longitude, latitude, bath):
        vertices_lon = [10, 10, -5, -10, 0, 0, -20, -20]
        vertices_lat = [65, 60, 59, 52.5, 47.5, 45, 40, 63]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath > 200) * (bath > 0) * (~xr.ufuncs.isnan(bath))
        return mask

    @classmethod
    def region_def_irish_sea(cls, longitude, latitude, bath):
        vertices_lon = [-5, -7.6, -7.5, -4.1, 0, -2.6]
        vertices_lat = [56.4, 55, 52, 50.7, 51.5, 55.3]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~xr.ufuncs.isnan(bath))
        return mask

    @classmethod
    def region_def_kattegat(cls, longitude, latitude, bath):
        vertices_lon = [9, 9, 13, 13]
        vertices_lat = [60, 52.5, 52.5, 60]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~xr.ufuncs.isnan(bath))
        return mask
