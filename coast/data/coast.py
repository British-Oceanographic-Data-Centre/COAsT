"""The coast class is the main access point into this package."""
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
import copy
from .._utils.logging_util import get_slug, debug, info, warn, warning
from .opendap import OpendapInfo


def setup_dask_client(workers: int = 2, threads: int = 2, memory_limit_per_worker: str = "2GB"):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class Coast:
    def __init__(
        self,
        file: str = None,
        chunks: dict = None,
        multiple: bool = False,
        workers: int = 2,  # TODO Do something with this unused parameter or delete it
        threads: int = 2,  # TODO Do something with this unused parameter or delete it
        memory_limit_per_worker: str = "2GB",  # TODO Do something with this unused parameter or delete it
    ):
        """This is the constructor method.

        This is the access point into the COAsT class. From here all the magic happens.

        Args:
            file:
            chunks: how/where to break up the data when reading it in
            multiple:
            workers: optional Dask related input
            threads: optional Dask related input
            memory_limit_per_worker: optional Dask related input
        """
        debug(f"Creating a new {get_slug(self)}")
        self.dataset = None
        self.earth_raids = 6371.007176  # Radius of the earth in km TODO Could this be module-level?
        self.set_dimension_mapping()
        self.set_variable_mapping()

        if file is None:
            warn(
                "Object created but no file or directory specified: \n"
                "{0} \n"
                "Use COAsT.load() to load a NetCDF file from file path or directory into this object.".format(
                    str(self)
                ),
                UserWarning,
            )
        else:
            self.load(file, chunks, multiple)
        debug(f"{get_slug(self)} initialised")

    def load(self, file_or_dir, chunks: dict = None, multiple=False):
        """
        Loads a file into a COAsT object's dataset variable using xarray

        Args:
            file_or_dir (str): file name, OPeNDAP accessor, or directory to multiple files.
            chunks (dict): Chunks to use in Dask [default None]
            multiple (bool): If true, load in multiple files from directory.
                             If false load a single file [default False]
        """
        if (opendap := isinstance(file_or_dir, OpendapInfo)) and multiple:
            raise NotImplementedError("Loading multiple OPeNDAP datasets is not supported")
        elif opendap:
            self.load_dataset(file_or_dir.open_dataset(chunks=chunks))
        elif multiple:
            self.load_multiple(file_or_dir, chunks)
        else:
            self.load_single(file_or_dir, chunks)

    def __getitem__(self, name: str):
        return self.dataset[name]

    def load_single(self, file, chunks: dict = None):
        """Loads a single file into COAsT object's dataset variable."""
        info(f"Loading a single file ({file} for {get_slug(self)}")
        self.dataset = xr.open_dataset(file, chunks=chunks)

    def load_multiple(self, directory_to_files, chunks: dict = None):
        """Loads multiple files from directory into dataset variable."""
        info(f"Loading a directory ({directory_to_files}) for {get_slug(self)}")
        self.dataset = xr.open_mfdataset(
            directory_to_files, chunks=chunks, parallel=True, combine="by_coords"
        )  # , compat='override')

    def load_dataset(self, dataset):
        """
        :param dataset: The dataset to use
        :type dataset: xarray.Dataset
        """
        self.dataset = dataset
        debug(f"Dataset for {get_slug(self)} set to {get_slug(dataset)}")

    def set_dimension_mapping(self):
        self.dim_mapping = None  # TODO Object attributes should be defined in the __init__
        debug(f"dim_mapping for {get_slug(self)} set to {self.dim_mapping}")

    def set_variable_mapping(self):
        self.var_mapping = None  # TODO Object attributes should be defined in the __init__
        debug(f"var_mapping for {get_slug(self)} set to {self.var_mapping}")

    def set_grid_ref_attribute(self):
        self.grid_ref_attr_mapping = None  # TODO Object attributes should be defined in the __init__
        debug(f"grid_ref_attr_mapping for {get_slug(self)} set to {self.grid_ref_attr_mapping}")

    def set_dimension_names(self, dim_mapping: dict):
        """
        Relabel dimensions in COAsT object xarray.dataset to ensure
        consistent naming throughout the COAsT package.

        Args:
            dim_mapping (dict): keys are dimension names to change and values
                                new dimension names
        """
        debug(f"Setting dimension names for {get_slug(self)} with mapping {dim_mapping}")
        if dim_mapping is None:
            return
        for key, value in dim_mapping.items():
            try:
                self.dataset = self.dataset.rename_dims({key: value})
            except ValueError as err:
                warning(
                    f"{get_slug(self)}: Problem renaming dimension from {get_slug(self.dataset)}: {key} -> {value}."
                    f"{chr(10)}Error message of '{err}'"
                )

    def set_variable_names(self, var_mapping: dict):
        """
        Relabel variables in COAsT object xarray.dataset to ensure
        consistent naming throughout the COAsT package.

        Args:
            var_mapping (dict): keys are variable names to change and values
                                are new variable names
        """
        debug(f"Setting variable names for {get_slug(self)} with mapping {var_mapping}")
        if var_mapping is None:
            return
        for key, value in var_mapping.items():
            try:
                self.dataset = self.dataset.rename_vars({key: value})
            except ValueError as err:
                warning(
                    f"{get_slug(self)}: Problem renaming variables from {get_slug(self.dataset)}: {key} -> {value}."
                    f"{chr(10)}Error message of '{err}'"
                )

    def set_variable_grid_ref_attribute(self, grid_ref_attr_mapping: dict):  # TODO is this still used?
        """
        Set attributes for variables to access within package.
        Set grid attributes to identify with grid variable is associated with.
        """
        debug(f"Setting variable attributes for {get_slug(self)} with mapping {grid_ref_attr_mapping}")
        if grid_ref_attr_mapping is None:
            return
        for key, value in grid_ref_attr_mapping.items():
            try:
                self.dataset[key].attrs["grid_ref"] = value
            except KeyError as err:
                warning(
                    f"{get_slug(self)}: Problem assigning attributes in {get_slug(self.dataset)}: {key} -> {value}."
                    f"{chr(10)}Error message of '{err}'"
                )

    def copy(self):
        new = copy.copy(self)
        debug(f"Copied {get_slug(self)} to new {get_slug(new)}")
        return new

    def isel(self, indexers: dict = None, drop: bool = False, **kwargs):
        """
        Indexes COAsT object along specified dimensions using xarray isel.
        Input is of same form as xarray.isel. Basic use, hand in either:
            1. dictionary with keys = dimensions, values = indices
            2. **kwargs of form dimension = indices
        """
        obj_copy = self.copy()
        debug(f"Indexing (isel) {get_slug(obj_copy)}")
        obj_copy.dataset = obj_copy.dataset.isel(indexers, drop, **kwargs)
        return obj_copy

    def sel(self, indexers: dict = None, drop: bool = False, **kwargs):
        """
        Indexes COAsT object along specified dimensions using xarray sel.
        Input is of same form as xarray.sel. Basic use, hand in either:
            1. Dictionary with keys = dimensions, values = indices
            2. **kwargs of form dimension = indices
        """
        obj_copy = self.copy()
        debug(f"Indexing (sel) {get_slug(obj_copy)}")
        obj_copy.dataset = obj_copy.dataset.sel(indexers, drop, **kwargs)
        return obj_copy

    def rename(self, rename_dict, **kwargs):
        debug(f"Renaming {get_slug(self.dataset)} with dict {rename_dict}")
        self.dataset = self.dataset.rename(rename_dict, **kwargs)
        return  # TODO Should this return something? If not, the statement is not needed

    def subset(self, **kwargs):
        """
        Subsets all variables within the dataset inside self (a COAsT object).
        Input is a set of keyword argument pairs of the form:
            dimension_name = indices
        The entire object is then subsetted along this dimension at indices
        """
        debug(f"Subsetting {get_slug(self)}")
        self.dataset = self.dataset.isel(kwargs)

    def subset_as_copy(self, **kwargs):
        """
        Similar to COAsT.subset() however applies the subsetting to a copy of
        the original COAsT object. This subsetted copy is then returned.
        Useful for preserving the original object whilst creating smaller
        subsetted object copies.
        """
        debug(f"Subsetting as copy {get_slug(self.dataset)}")
        obj_copy = self.copy()
        obj_copy.subset(**kwargs)
        return obj_copy

    def distance_between_two_points(self):
        raise NotImplementedError  # TODO Should this class be decorated as an abstractclass?

    def subset_indices_by_distance(self, centre_lon: float, centre_lat: float, radius: float):
        """
        This method returns a `tuple` of indices within the `radius` of the lon/lat point given by the user.

        Distance is calculated as haversine - see `self.calculate_haversine_distance`

        :param centre_lon: The longitude of the users central point
        :param centre_lat: The latitude of the users central point
        :param radius: The haversine distance (in km) from the central point
        :return: All indices in a `tuple` with the haversine distance of the central point
        """
        debug(f"Subsetting {self} indices by distance")
        # Flatten NEMO domain stuff.
        lon = self.dataset.longitude
        lat = self.dataset.latitude

        # Calculate the distances between every model point and the specified
        # centre. Calls another routine dist_haversine.

        dist = self.calculate_haversine_distance(centre_lon, centre_lat, lon, lat)
        indices_bool = dist < radius
        indices = np.where(indices_bool.compute())

        return xr.DataArray(indices[0]), xr.DataArray(indices[1])

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]

        return: Indices corresponding to datapoints inside specified box
        """
        debug(f"Subsetting {get_slug(self)} indices within lon/lat")
        lon_str = "longitude"
        lat_str = "latitude"
        lon = self.dataset[lon_str].copy()  # TODO Add a comment explaining why this needs to be copied
        lat = self.dataset[lat_str]
        ff = lon > lonbounds[0]
        ff *= lon < lonbounds[1]
        ff *= lat > latbounds[0]
        ff *= lat < latbounds[1]

        return np.where(ff)

    @staticmethod
    def calculate_haversine_distance(lon1, lat1, lon2, lat2):  # TODO This could be a static method
        """
        # Estimation of geographical distance using the Haversine function.
        # Input can be single values or 1D arrays of locations. This
        # does NOT create a distance matrix but outputs another 1D array.
        # This works for either location vectors of equal length OR a single loc
        # and an arbitrary length location vector.
        #
        # lon1, lat1 :: Location(s) 1.
        # lon2, lat2 :: Location(s) 2.
        """

        debug(f"Calculating haversine distance between {lon1},{lat1} and {lon2},{lat2}")

        # Convert to radians for calculations
        lon1 = np.deg2rad(lon1)
        lat1 = np.deg2rad(lat1)
        lon2 = np.deg2rad(lon2)
        lat2 = np.deg2rad(lat2)

        # Latitude and longitude differences
        dlat = (lat2 - lat1) / 2
        dlon = (lon2 - lon1) / 2

        # Haversine function.
        distance = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
        distance = 2 * 6371.007176 * np.arcsin(np.sqrt(distance))

        return distance

    def get_subset_as_xarray(
        self, var: str, points_x: slice, points_y: slice, line_length: int = None, time_counter: int = 0
    ):
        """
        This method gets a subset of the data across the x/y indices given for the chosen variable.

        Setting time_counter to None will treat `var` as only having 3 dimensions depth, y, x

        there is a check on `var` to see the size of the time_counter, if 1 then time_counter is fixed to index 0.

        :param var: the name of the variable to get data from
        :param points_x: a list/array of indices for the x dimension
        :param points_y: a list/array of indices for the y dimension
        :param line_length: (Optional) the length of your subset (assuming simple line transect)  TODO This is unsued
        :param time_counter: (Optional) which time slice to get data from, if None and the variable only has one a time
                             channel of length 1 then time_counter is fixed too an index of 0
        :return: data across all depths for the chosen variable along the given indices
        """
        debug(f"Subsetting {var} from {get_slug(self)}")
        try:
            [time_size, _, _, _] = self.dataset[var].shape
            if time_size == 1:
                time_counter == 0  # TODO This should probably be =, not ==

        except ValueError:
            time_counter = None

        dx = xr.DataArray(points_x)
        dy = xr.DataArray(points_y)

        if time_counter is None:
            smaller = self.dataset[var].isel(x_dim=dx, y_dim=dy)
        else:
            smaller = self.dataset[var].isel(t_dim=time_counter, x_dim=dx, y_dim=dy)

        return smaller

    def get_2d_subset_as_xarray(
        self, var: str, points_x: slice, points_y: slice, line_length: int = None, time_counter: int = 0
    ):
        """

        :param var:
        :param points_x:
        :param points_y:
        :param line_length:
        :param time_counter:
        :return:
        """

        debug(f"Fetching {var} subset as xarray")
        try:
            [time_size, _, _, _] = self.dataset[var].shape
            if time_size == 1:
                time_counter == 0  # TODO This should probably be =, not ==
        except ValueError:
            time_counter = None

        if time_counter is None:
            smaller = self.dataset[var].isel(x=points_x, y=points_y)
        else:
            smaller = self.dataset[var].isel(time_counter=time_counter, x=points_x, y=points_y)

        return smaller

    def plot_simple_2d(self, x, y, data: xr.DataArray, cmap, plot_info: dict):
        """
        This is a simple method that will plot data in a 2d. It is a wrapper for matplotlib's 'pcolormesh' method.

        `cmap` and `plot_info` are required to run this method, `cmap` is passed directly to `pcolormesh`.

        `plot_info` contains all the required information for setting the figure;
         - ylim
         - xlim
         - clim
         - title
         - fig_size
         - ylabel

        :param x: The variable contain the x axis information
        :param y: The variable contain the y axis information
        :param data: the DataArray a user wishes to plot
        :param cmap:
        :param plot_info:
        :return:
        """
        info("Generating simple 2D plot...")
        import matplotlib.pyplot as plt

        plt.close("all")

        fig = plt.figure()
        plt.rcParams["figure.figsize"] = plot_info["fig_size"]

        fig.add_subplot(411)
        plt.pcolormesh(x, y, data, cmap=cmap)

        plt.ylim(plot_info["ylim"])
        plt.xlim(plot_info["xlim"])
        plt.title(plot_info["title"])
        plt.ylabel(plot_info["ylabel"])
        plt.clim(plot_info["clim"])
        plt.colorbar()

        return plt

    def plot_cartopy(self, var: str, plot_var: array, params, time_counter: int = 0):
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        except ImportError:
            import sys

            warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
            sys.exit(-1)

        import matplotlib.pyplot as plt

        info("Generating CartoPy plot...")
        plt.close("all")
        fig = plt.figure(figsize=(10, 10))
        fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

        cset = (
            self.dataset[var]
            .isel(time_counter=time_counter, deptht=0)
            .plot.pcolormesh(
                np.ma.masked_where(plot_var == np.NaN, plot_var), transform=ccrs.PlateCarree(), cmap=params.cmap
            )
        )

        cset.set_clim([params.levs[0], params.levs[-1]])

        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
        ax.add_feature(cartopy.feature.RIVERS)
        coast = NaturalEarthFeature(category="physical", scale="10m", facecolor="none", name="coastline")
        ax.add_feature(coast, edgecolor="gray")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="-"
        )

        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_right = False
        gl.ylabels_left = True
        gl.x_formatter = LONGITUDE_FORMATTER
        gl.y_formatter = LATITUDE_FORMATTER

        plt.colorbar(cset, shrink=params.colorbar_shrink, pad=0.05)

        # tmp = self.dataset.votemper
        # tmp.attrs = self.dataset.votemper.attrs
        # tmp.isel(time_counter=time_counter, deptht=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree())
        # ax.set_global()
        # ax.coastlines()
        info("Displaying plot!")
        plt.show()

    def plot_movie(self):
        raise NotImplementedError
