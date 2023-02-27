"""Mask maker"""
import xarray as xr
import numpy as np
import skimage.draw as draw
from . import general_utils


class MaskMaker:
    """
    MaskMasker is a class of methods to assist with making regional masks within COAsT.
    Presently these masks are external to MaskMaker.
    It constructs a gridded boolean numpy array for each region, which are stacked over a dim_mask dimension and
    stored as an xarray object.

    A typical workflow might be:

        # Define vertices
        vertices_lon = [-5, -5, 5, 5]
        vertices_lat = [40, 60, 60, 40]

        # input lat/lon as xr.DataArray or numpy arrays. Return gridded boolean mask np.array on target grid
        filled = mm.make_region_from_vertices(
            sci.dataset.longitude, sci.dataset.latitude, vertices_lon, vertices_lat)

        # make xr.Dataset of masks from gridded mask array or list of mask arrays
        gridded_mask = mm.make_mask_dataset(sci.dataset.longitude.values,
                                         sci.dataset.latitude.values,
                                         filled)
        # quick plot
        mm.quick_plot(gridded_mask)


    TO DO:
    * Sort out region naming to be consistently applied and associated with the masks E.g. defined regions, or user defined masks
    * Create final mask as a xr.DataArray, not a xr.Dataset
    """

    def __init__(self):
        return

    @staticmethod
    def make_mask_dataset(longitude, latitude, mask_list, mask_names: list = None):
        """
        create xr.Dataset for mask with latitude and longitude coordinates. If mask_names are given
        create a dim_mask coordinate of names

        """
        if type(mask_list) is not list:
            mask_list = [mask_list]
        gridded_mask = xr.Dataset()
        gridded_mask["longitude"] = (["y_dim", "x_dim"], longitude)
        gridded_mask["latitude"] = (["y_dim", "x_dim"], latitude)

        if mask_names is not None:
            gridded_mask["region_names"] = (["dim_mask"], mask_names)
        else:
            gridded_mask["region_names"] = (["dim_mask"], range(len(mask_list)))

        gridded_mask = gridded_mask.set_coords(["longitude", "latitude", "region_names"])
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
        Filled 2D np.array
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
    def region_def_nws_north_north_sea(cls, longitude, latitude, bath):
        """
        Regional definition for the northern North Sea (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [-5.34, -0.7, 7.5, 7.5, 9, 9, 6.3, 6.3, 5, 5, 4.126, 4.126, -1.071]
        vertices_lat = [56.93, 54.09, 54.09, 56, 56, 57.859, 57.859, 58.121, 58.121, 58.59, 58.59, 60.5, 60.5]

        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_south_north_sea(cls, longitude, latitude, bath):
        vertices_lon = [-0.67, -0.67, 9, 9, 7.57, 7.57]
        vertices_lat = [54.08, 51, 51, 56, 56, 54.08]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_outer_shelf(cls, longitude, latitude, bath):
        """
        Regional definition for the Outer Shelf (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [-4.1, -9.5, -1, 3.171, 3.171, -3.76, -3.76, -12, -12, -12, -4]
        vertices_lat = [50.7, 52.71, 60.5, 60.45, 63.3, 63.3, 60.45, 60.45, 55.28, 48, 48]
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
        vertices_lon = [-3.99, -3.99, -3.5, 12, 9]
        vertices_lat = [51, 48.8, 48, 48, 51]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_off_shelf(cls, longitude, latitude, bath):
        vertices_lon = [10, 10, -5, -10, 0, 0, -20, -20]
        vertices_lat = [65, 60, 59, 52.5, 47.5, 45, 40, 63]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath > 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_irish_sea(cls, longitude, latitude, bath):
        vertices_lon = [-5, -7.6, -7.5, -4.1, 0, -2.6]
        vertices_lat = [56.4, 55, 52, 50.7, 51.5, 55.3]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def region_def_nws_kattegat(cls, longitude, latitude, bath):
        vertices_lon = [9, 9, 13, 13]
        vertices_lat = [60, 52.5, 52.5, 60]
        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath < 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    def region_def_nws_fsc(cls, longitude, latitude, bath):
        """
        Regional definition for Faroe Shetland Channel (Northwest European Shelf)
        Longitude, latitude and bath should be 2D arrays corresponding to model
        coordinates and bathymetry. Bath should be positive with depth.
        """
        vertices_lon = [-7.13, -9.72, -6.37, -0.45, -4.53]
        vertices_lat = [62.17, 60.6, 59.07, 61.945, 62.51]

        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        mask = mask * (bath > 200) * (bath > 0) * (~np.isnan(bath))
        return mask

    @classmethod
    def make_region_from_vertices(cls, longitude, latitude, vertices_lon: list, vertices_lat: list):
        """
        Construct mask on supplied longitude, latitude grid with input lists of lon and lat polygon vertices
        :param longitude: np.array/xr.DataArray of longitudes on target grid
        :param latitude: np.array/xr.DataArray of latitudes on target grid
        :param vertices_lon: list of vertices for bounding polygon
        :param vertices_lat: list of vertices for bounding polygon
        :return: mask: np.array(boolean) on target grid. Ones are bound by polygon vertices
        """
        try:
            longitude = longitude.values
            latitude = longitude.values
        except AttributeError:
            pass

        mask = cls.fill_polygon_by_lonlat(np.zeros(longitude.shape), longitude, latitude, vertices_lon, vertices_lat)
        return mask

    @classmethod
    def quick_plot(cls, mask: xr.Dataset):
        """
        Plot a map of masks in the MaskMaker object
        Add labels
        """
        import matplotlib.pyplot as plt

        n_mask = mask.dims["dim_mask"]
        offset = 10  # nonzero offset to make scaled-boolean-masks [0, >offset]
        for j in range(0, n_mask, 1):
            tt = (j + offset) * mask["mask"].isel(dim_mask=j).squeeze()
            ff = tt.where(tt > 0).plot(
                x="longitude", y="latitude", levels=range(offset, n_mask + offset + 1, 1), add_colorbar=False
            )

        cbar = plt.colorbar(ff)
        cbar.ax.get_yaxis().set_ticks([])
        for j in range(0, n_mask, 1):
            cbar.ax.text(
                1 + 0.5,
                offset + (j + 0.5),
                mask["region_names"].isel(dim_mask=j).values,
                ha="left",
                va="center",
                color="red",
            )
        cbar.ax.get_yaxis().labelpad = 15
        plt.title(None)
