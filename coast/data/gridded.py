"""Gridded class"""
import os.path as path_lib
import warnings

# from dask import delayed, compute, visualize
# import graphviz
import gsw
import numpy as np
import xarray as xr

from .._utils import general_utils, stats_util
from .coast import Coast
from .config_parser import ConfigParser
from .._utils.logging_util import get_slug, debug, info, warn, error, warning
import pandas as pd


class Gridded(Coast):  # TODO Complete this docstring
    """
    Words to describe the NEMO class

    kwargs -- define addition keyworded arguemts for domain file. E.g. ln_sco=1
    if using s-scoord in an old domain file that does not carry this flag.
    """

    def __init__(
        self,
        fn_data=None,
        fn_domain=None,  # TODO Super init not called + add a docstring
        multiple=False,
        config: str = " ",
        workers=2,
        threads=2,
        memory_limit_per_worker="2GB",
        **kwargs,
    ):
        debug(f"Creating new {get_slug(self)}")
        self.dataset = xr.Dataset()
        self.grid_ref = None
        self.domain_loaded = False
        self.fn_data = fn_data
        self.fn_domain = fn_domain
        self.grid_vars = None

        if path_lib.isfile(config):
            self.config = ConfigParser(config).config
            if self.config.chunks:
                self._setup_grid_obj(self.config.chunks, multiple, **kwargs)
            else:
                self._setup_grid_obj(None, multiple, **kwargs)
        else:  # allow for usage without config file, this will be limted and dosen't bring the full COAST features
            debug("Config file expected. Limited functionality without config file")
            if self.fn_data is not None:
                self.load(self.fn_data, None, multiple)
            if self.fn_domain is not None:
                self.filename_domain = self.fn_domain
                dataset_domain = self.load_domain(self.fn_domain, None)
                self.dataset["domain"] = dataset_domain

    def _setup_grid_obj(self, chunks, multiple, **kwargs):
        """This is a helper method to reduce the size of def __init__

        Args:
            chunks: This is a setting for xarray as to whether dask (parrell processing) should be on and how it works
            multiple: falg to tell if we are loading one or more files
            **kwargs: pass direct to loaded xarray dataset
        """
        self.set_grid_vars()
        self.set_dimension_mapping()
        self.set_variable_mapping()

        if self.fn_data is not None:
            self.load(self.fn_data, chunks, multiple)

        self.set_dimension_names(self.config.dataset.dimension_map)
        self.set_variable_names(self.config.dataset.variable_map)

        if self.fn_domain is None:
            self.filename_domain = ""  # empty store for domain fileanme
            warn("No NEMO domain specified, only limited functionality" + " will be available")
        else:
            self.filename_domain = self.fn_domain  # store domain fileanme
            dataset_domain = self.load_domain(self.fn_domain, chunks)

            # Define extra domain attributes using kwargs dictionary
            # This is a bit of a placeholder. Some domain/nemo files will have missing variables
            for key, value in kwargs.items():
                dataset_domain[key] = value

            if self.fn_data is not None:
                dataset_domain = self.trim_domain_size(dataset_domain)
            self.set_timezero_depths(
                dataset_domain
            )  # THIS ADDS TO dataset_domain. Should it be 'return'ed (as in trim_domain_size) or is implicit OK?
            self.merge_domain_into_dataset(dataset_domain)
            debug(f"Initialised {get_slug(self)}")

    def set_grid_vars(self):
        """Define the variables to map from the domain file to the NEMO obj"""
        # Define grid specific variables to pull across
        #
        for key, value in self.config.grid_ref.items():
            self.grid_ref = key
            self.grid_vars = value

    # TODO Add parameter type hints and a docstring
    def load_domain(self, fn_domain, chunks):  # TODO Do something with this unused parameter or remove it
        """Loads domain file and renames dimensions with dim_mapping_domain"""
        # Load xarray dataset
        info(f'Loading domain: "{fn_domain}"')
        dataset_domain = xr.open_dataset(fn_domain)
        self.domain_loaded = True
        # Rename dimensions
        for key, value in self.config.domain.dimension_map.items():
            mapping = {key: value}
            try:
                dataset_domain = dataset_domain.rename_dims(mapping)
            except ValueError as err:
                warning(
                    f"{get_slug(self)}: Problem renaming dimension from {get_slug(self.dataset)}: {key} -> {value}."
                    f"{chr(10)}Error message of '{err}'"
                )

        return dataset_domain

    def merge_domain_into_dataset(self, dataset_domain):
        """Merge domain dataset variables into self.dataset, using grid_ref"""
        debug(f"Merging {get_slug(dataset_domain)} into {get_slug(self)}")
        # Define grid independent variables to pull across

        all_vars = self.grid_vars + self.config.code_processing.not_grid_variables

        # Trim domain DataArray area if necessary.
        self.copy_domain_vars_to_dataset(dataset_domain, self.grid_vars)

        # Reset & set specified coordinates
        self.dataset = self.dataset.reset_coords()
        for var in self.config.dataset.coord_var:
            try:
                self.dataset = self.dataset.set_coords(var)
            except ValueError as err:
                warning(f"Issue with settings coordinates using value {var}.{chr(10)}Error message of {err}")

        # Delete specified variables
        for var in self.config.code_processing.delete_variables:
            try:
                self.dataset = self.dataset.drop(var)
            except ValueError as err:
                warning(f"Issue with dropping variable {var}.{chr(10)}Error message of {err}")

    def __getitem__(self, name: str):
        return self.dataset[name]

    def set_grid_ref_attr(self):  # possible not used
        debug(f"{get_slug(self)} grid_ref_attr set to {self.grid_ref_attr_mapping}")
        self.grid_ref_attr_mapping = {
            "temperature": "t-grid",
            "coast_name_for_u_velocity": "u-grid",
            "coast_name_for_v_velocity": "v-grid",
            "coast_name_for_w_velocity": "w-grid",
            "coast_name_for_vorticity": "f-grid",
        }

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        debug(f"Fetching contour complex from {get_slug(self)}")
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method="nearest", tolerance=tolerance)
        return smaller

    def set_timezero_depths(self, dataset_domain):
        """
        Calculates the depths at time zero (from the domain_cfg input file)
        for the appropriate grid.
        The depths are assigned to domain_dataset.depth_0
        """
        debug(f"Setting timezero depths for {get_slug(self)} with {get_slug(dataset_domain)}")

        try:
            bathymetry = dataset_domain.bathy_metry.squeeze()
        except AttributeError as err:
            bathymetry = xr.zeros_like(dataset_domain.e1t.squeeze())
            (
                warnings.warn(
                    f"The model domain loaded, '{self.filename_domain}', does not contain the "
                    "bathy_metry' variable. This will result in the "
                    "NEMO.dataset.bathymetry variable being set to zero, which "
                    "may result in unexpected behaviour from routines that require "
                    "this variable."
                )
            )
            debug(
                f"The bathy_metry variable was missing from the domain_cfg for "
                f"{get_slug(self)} with {get_slug(dataset_domain)}"
                f"{chr(10)}Error message of {err}"
            )
        try:
            if self.grid_ref == "t-grid":
                e3w_0 = np.squeeze(dataset_domain.e3w_0.values)
                depth_0 = np.zeros_like(e3w_0)
                depth_0[0, :, :] = 0.5 * e3w_0[0, :, :]
                depth_0[1:, :, :] = depth_0[0, :, :] + np.cumsum(e3w_0[1:, :, :], axis=0)
            elif self.grid_ref == "w-grid":
                e3t_0 = np.squeeze(dataset_domain.e3t_0.values)
                depth_0 = np.zeros_like(e3t_0)
                depth_0[0, :, :] = 0.0
                depth_0[1:, :, :] = np.cumsum(e3t_0, axis=0)[:-1, :, :]
            elif self.grid_ref == "u-grid":
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_u = 0.5 * (e3w_0[:, :, :-1] + e3w_0[:, :, 1:])
                depth_0 = np.zeros_like(e3w_0)
                depth_0[0, :, :-1] = 0.5 * e3w_0_on_u[0, :, :]
                depth_0[1:, :, :-1] = depth_0[0, :, :-1] + np.cumsum(e3w_0_on_u[1:, :, :], axis=0)
                bathymetry[:, :-1] = 0.5 * (bathymetry[:, :-1] + bathymetry[:, 1:])
            elif self.grid_ref == "v-grid":
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_v = 0.5 * (e3w_0[:, :-1, :] + e3w_0[:, 1:, :])
                depth_0 = np.zeros_like(e3w_0)
                depth_0[0, :-1, :] = 0.5 * e3w_0_on_v[0, :, :]
                depth_0[1:, :-1, :] = depth_0[0, :-1, :] + np.cumsum(e3w_0_on_v[1:, :, :], axis=0)
                bathymetry[:-1, :] = 0.5 * (bathymetry[:-1, :] + bathymetry[1:, :])
            elif self.grid_ref == "f-grid":
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_f = 0.25 * (e3w_0[:, :-1, :-1] + e3w_0[:, :-1, 1:] + e3w_0[:, 1:, :-1] + e3w_0[:, 1:, 1:])
                depth_0 = np.zeros_like(e3w_0)
                depth_0[0, :-1, :-1] = 0.5 * e3w_0_on_f[0, :, :]
                depth_0[1:, :-1, :-1] = depth_0[0, :-1, :-1] + np.cumsum(e3w_0_on_f[1:, :, :], axis=0)
                bathymetry[:-1, :-1] = 0.25 * (
                    bathymetry[:-1, :-1] + bathymetry[:-1, 1:] + bathymetry[1:, :-1] + bathymetry[1:, 1:]
                )
            else:
                raise ValueError(str(self) + ": " + self.grid_ref + " depth calculation not implemented")
            # Write the depth_0 variable to the domain_dataset DataSet, with grid type
            dataset_domain[f"depth{self.grid_ref.replace('-grid', '')}_0"] = xr.DataArray(
                depth_0,
                dims=["z_dim", "y_dim", "x_dim"],
                attrs={"units": "m", "standard_name": "Depth at time zero on the {}".format(self.grid_ref)},
            )

            self.dataset["bathymetry"] = bathymetry
            self.dataset["bathymetry"].attrs = {
                "units": "m",
                "standard_name": "bathymetry",
                "description": "depth of last wet w-level on the horizontal {}".format(self.grid_ref),
            }
        except ValueError as err:
            error(err)

    # Add subset method to NEMO class
    def subset_indices(self, *, start: tuple, end: tuple) -> tuple:
        """
        based on transect_indices, this method looks to return all indices between the given points.
        This results in a 'box' (Quadrilateral) of indices.
        consequently the returned lists may have different lengths.
        :param start: A lat/lon pair
        :param end: A lat/lon pair
        :return: list of y indices, list of x indices,
        """
        debug(f"Subsetting {get_slug(self)} indices from {start} to {end}")
        [j1, i1] = self.find_j_i(lat=start[0], lon=start[1])  # lat , lon
        [j2, i2] = self.find_j_i(lat=end[0], lon=end[1])  # lat , lon

        return list(np.arange(j1, j2 + 1)), list(np.arange(i1, i2 + 1))

    def find_j_i(self, *, lat: float, lon: float):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(lat=49, lon=-12)

        :param lat: latitude
        :param lon: longitude
        :return: the y and x coordinates for the NEMO object's grid_ref, i.e. t,u,v,f,w.
        """
        debug(f"Finding j,i for {lat},{lon} from {get_slug(self)}")
        dist2 = np.square(self.dataset.latitude - lat) + np.square(self.dataset.longitude - lon)
        [y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def find_j_i_domain(self, *, lat: float, lon: float, dataset_domain: xr.DataArray):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i_domain(lat=49, lon=-12, dataset_domain=dataset_domain)

        :param lat: latitude
        :param lon: longitude
        :param dataset_domain: dataset domain
        :return: the y and x coordinates for the grid_ref variable within the domain file
        """
        debug(f"Finding j,i domain for {lat},{lon} from {get_slug(self)} using {get_slug(dataset_domain)}")
        internal_lat = dataset_domain[self.grid_vars[1]]  # [f"gphi{self.grid_ref.replace('-grid','')}"]
        internal_lon = dataset_domain[self.grid_vars[0]]  # [f"glam{self.grid_ref.replace('-grid','')}"]
        dist2 = np.square(internal_lat - lat) + np.square(internal_lon - lon)
        [_, y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def transect_indices(self, start: tuple, end: tuple) -> tuple:
        """
        This method returns the indices of a simple straight line transect between two
        lat lon points defined on the NEMO object's grid_ref, i.e. t,u,v,f,w.

        :type start: tuple A lat/lon pair
        :type end: tuple A lat/lon pair
        :return: array of y indices, array of x indices, number of indices in transect
        """
        debug(f"Fetching transect indices for {start} to {end} from {get_slug(self)}")
        [j1, i1] = self.find_j_i(lat=start[0], lon=start[1])  # lat , lon
        [j2, i2] = self.find_j_i(lat=end[0], lon=end[1])  # lat , lon

        line_length = max(np.abs(j2 - j1), np.abs(i2 - i1)) + 1

        jj1 = [int(jj) for jj in np.round(np.linspace(j1, j2, num=line_length))]
        ii1 = [int(ii) for ii in np.round(np.linspace(i1, i2, num=line_length))]

        return jj1, ii1, line_length

    @staticmethod
    def interpolate_in_space(model_array, new_lon, new_lat, mask=None):
        """
        Interpolates a provided xarray.DataArray in space to new longitudes
        and latitudes using a nearest neighbour method (BallTree).

        Example Usage
        ----------
        # Get an interpolated DataArray for temperature onto two locations
        interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                                 [0,1], [45,46])
        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_lons (1Darray): Array of longitudes (degrees) to compare with model
        new_lats (1Darray): Array of latitudes (degrees) to compare with model
        mask (2D array): Mask array. Where True (or 1), elements of array will
                     not be included. For example, use to mask out land in
                     case it ends up as the nearest point.

        Returns
        -------
        Interpolated DataArray
        """
        debug(f"Interpolating {get_slug(model_array)} in space with nearest neighbour")
        # Get nearest indices
        ind_x, ind_y = general_utils.nearest_indices_2d(
            model_array.longitude, model_array.latitude, new_lon, new_lat, mask=mask
        )

        # Geographical interpolation (using BallTree indices)
        interpolated = model_array.isel(x_dim=ind_x, y_dim=ind_y)
        if "dim_0" in interpolated.dims:
            interpolated = interpolated.rename({"dim_0": "interp_dim"})
        return interpolated

    @staticmethod
    def interpolate_in_time(model_array, new_times, interp_method="nearest", extrapolate=True):
        """
        Interpolates a provided xarray.DataArray in time to new python
        datetimes using a specified scipy.interpolate method.

        Example Useage
        ----------
        # Get an interpolated DataArray for temperature onto altimetry times
        new_times = altimetry.dataset.time
        interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                                 new_times)
        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_times (array): New times to interpolate to (array of datetimes)
        interp_method (str): Interpolation method

        Returns
        -------
        Interpolated DataArray
        """
        debug(f'Interpolating {get_slug(model_array)} in time with method "{interp_method}"')
        # Time interpolation
        interpolated = model_array.swap_dims({"t_dim": "time"})
        if extrapolate:
            interpolated = interpolated.interp(
                time=new_times, method=interp_method, kwargs={"fill_value": "extrapolate"}
            )
        else:
            interpolated = interpolated.interp(time=new_times, method=interp_method)
        # interpolated = interpolated.swap_dims({'time':'t_dim'})  # TODO Do something with this or delete it

        return interpolated

    def construct_density(self, eos="EOS10"):

        """
            Constructs the in-situ density using the salinity, temperture and
            depth_0 fields and adds a density attribute to the t-grid dataset

            Requirements: The supplied t-grid dataset must contain the
            Practical Salinity and the Potential Temperature variables. The depth_0
            field must also be supplied. The GSW package is used to calculate
            The Absolute Pressure, Absolute Salinity and Conservate Temperature.

            Note that currently density can only be constructed using the EOS10
            equation of state.

        Parameters
        ----------
        eos : equation of state, optional
            DESCRIPTION. The default is 'EOS10'.


        Returns
        -------
        None.
        adds attribute NEMO.dataset.density

        """
        debug(f'Constructing in-situ density for {get_slug(self)} with EOS "{eos}"')
        try:
            if eos != "EOS10":
                raise ValueError(str(self) + ": Density calculation for " + eos + " not implemented.")
            if self.grid_ref != "t-grid":
                raise ValueError(
                    str(self)
                    + ": Density calculation can only be performed for a t-grid object,\
                                 the tracer grid for NEMO."
                )

            try:
                shape_ds = (
                    self.dataset.t_dim.size,
                    self.dataset.z_dim.size,
                    self.dataset.y_dim.size,
                    self.dataset.x_dim.size,
                )
                sal = self.dataset.salinity.to_masked_array()
                temp = self.dataset.temperature.to_masked_array()
            except AttributeError:
                shape_ds = (1, self.dataset.z_dim.size, self.dataset.y_dim.size, self.dataset.x_dim.size)
                sal = self.dataset.salinity.to_masked_array()[np.newaxis, ...]
                temp = self.dataset.temperature.to_masked_array()[np.newaxis, ...]

            density = np.ma.zeros(shape_ds)

            s_levels = self.dataset.depth_0.to_masked_array()
            lat = self.dataset.latitude.values
            lon = self.dataset.longitude.values
            # Absolute Pressure
            pressure_absolute = np.ma.masked_invalid(gsw.p_from_z(-s_levels, lat))  # depth must be negative
            # Absolute Salinity
            sal_absolute = np.ma.masked_invalid(gsw.SA_from_SP(sal, pressure_absolute, lon, lat))
            sal_absolute = np.ma.masked_less(sal_absolute, 0)
            # Conservative Temperature
            temp_conservative = np.ma.masked_invalid(gsw.CT_from_pt(sal_absolute, temp))
            # In-situ density
            density = np.ma.masked_invalid(gsw.rho(sal_absolute, temp_conservative, pressure_absolute))

            coords = {
                "depth_0": (("z_dim", "y_dim", "x_dim"), self.dataset.depth_0.values),
                "latitude": (("y_dim", "x_dim"), self.dataset.latitude.values),
                "longitude": (("y_dim", "x_dim"), self.dataset.longitude.values),
            }
            dims = ["z_dim", "y_dim", "x_dim"]
            attributes = {"units": "kg / m^3", "standard name": "In-situ density"}

            if shape_ds[0] != 1:
                coords["time"] = (("t_dim"), self.dataset.time.values)
                dims.insert(0, "t_dim")

            self.dataset["density"] = xr.DataArray(np.squeeze(density), coords=coords, dims=dims, attrs=attributes)

        except AttributeError as err:
            error(err)

    def trim_domain_size(self, dataset_domain):
        """
        Trim the domain variables if the dataset object is a spatial subset

        Note: This breaks if the SW & NW corner values of nav_lat and nav_lon
        are masked, as can happen if on land...
        """
        debug(f"Trimming {get_slug(self)} variables with {get_slug(dataset_domain)}")
        if (self.dataset["x_dim"].size != dataset_domain["x_dim"].size) or (
            self.dataset["y_dim"].size != dataset_domain["y_dim"].size
        ):
            info(
                "The domain  and dataset objects are different sizes:"
                " [{},{}] cf [{},{}]. Trim domain.".format(
                    dataset_domain["x_dim"].size,
                    dataset_domain["y_dim"].size,
                    self.dataset["x_dim"].size,
                    self.dataset["y_dim"].size,
                )
            )

            # Find the corners of the cut out domain.
            [j0, i0] = self.find_j_i_domain(
                lat=self.dataset.nav_lat[0, 0], lon=self.dataset.nav_lon[0, 0], dataset_domain=dataset_domain
            )
            [j1, i1] = self.find_j_i_domain(
                lat=self.dataset.nav_lat[-1, -1], lon=self.dataset.nav_lon[-1, -1], dataset_domain=dataset_domain
            )

            dataset_subdomain = dataset_domain.isel(y_dim=slice(j0, j1 + 1), x_dim=slice(i0, i1 + 1))
            return dataset_subdomain
        else:
            return dataset_domain

    def copy_domain_vars_to_dataset(self, dataset_domain, grid_vars):
        """
        Map the domain coordand metric variables to the dataset object.
        Expects the source and target DataArrays to be same sizes.
        """
        debug(f"Copying domain vars from {get_slug(dataset_domain)}/{get_slug(grid_vars)} to {get_slug(self)}")
        for var in grid_vars:
            try:
                new_name = self.config.domain.variable_map[var]
                self.dataset[new_name] = dataset_domain[var].squeeze()
                debug("map: {} --> {}".format(var, new_name))
            except:  # FIXME Catch specific exception(s)
                pass  # TODO Should we log something here?

    def differentiate(self, in_var_str, config_path=None, dim="z_dim", out_var_str=None, out_obj=None):
        """
        Derivatives are computed in x_dim, y_dim, z_dim (or i,j,k) directions
        wrt lambda, phi, or z coordinates (with scale factor in metres not degrees).

        Derivatives are calculated using the approach adopted in NEMO,
        specifically using the 1st order accurate central difference
        approximation. For reference see section 3.1.2 (sec. Discrete operators)
        of the NEMO v4 Handbook.

        Currently the method does not accomodate all possible eventualities. It
        covers:
        1) d(grid_t)/dz --> grid_w

        Returns  an object (with the appropriate target grid_ref) containing
        derivative (out_var_str) as xr.DataArray

        This is hardwired to expect:
        1) depth_0 and e3_0 fields exist
        2) xr.DataArrays are 4D
        3) self.filename_domain if out_obj not specified
        4) If out_obj is not specified, one is built that is  the size of
            self.filename_domain. I.e. automatic subsetting of out_obj is not
            supported.

        Example usage:
        --------------
        # Initialise DataArrays
        nemo_t = coast.NEMO( fn_data, fn_domain, grid_ref='t-grid' )
        # Compute dT/dz
        nemo_w_1 = nemo_t.differentiate( 'temperature', dim='z_dim' )

        # For f(z)=-z. Compute df/dz = -1. Surface value is set to zero
        nemo_t.dataset['depth4D'],_ = xr.broadcast( nemo_t.dataset['depth_0'], nemo_t.dataset['temperature'] )
        nemo_w_4 = nemo_t.differentiate( 'depth4D', dim='z_dim', out_var_str='dzdz' )

        Provide an existing target NEMO object and target variable name:
        nemo_w_1 = nemo_t.differentiate( 'temperature', dim='z_dim', out_var_str='dTdz', out_obj=nemo_w_1 )


        Parameters
        ----------
        in_var_str : str, name of variable to differentiate
        config_path : str, path to the w grid config file
        dim : str, dimension to operate over. E.g. {'z_dim', 'y_dim', 'x_dim', 't_dim'}
        out_var_str : str, (optional) name of the target xr.DataArray
        out_obj : exiting NEMO obj to store xr.DataArray (optional)

        """
        import xarray as xr

        new_units = ""

        # Check in_var_str exists in self.
        if hasattr(self.dataset, in_var_str):
            # self.dataset[in_var_str] exists

            var = self.dataset[in_var_str]  # for convenience

            nt = var.sizes["t_dim"]
            nz = var.sizes["z_dim"]
            ny = var.sizes["y_dim"]
            nx = var.sizes["x_dim"]

            # Compute d(t_grid)/dz --> w-grid
            # Check grid_ref and dir. Determine target grid_ref.
            if (self.grid_ref == "t-grid") and (dim == "z_dim"):
                out_grid = "w-grid"

                # If out_obj exists check grid_ref, else create out_obj.
                if (out_obj is None) or (out_obj.grid_ref != out_grid):
                    try:
                        out_obj = Gridded(fn_domain=self.filename_domain, config=config_path)
                    except:  # TODO Catch specific exception(s)
                        warn(
                            "Failed to create target NEMO obj. Perhaps self.",
                            "filename_domain={} is empty?".format(self.filename_domain),
                        )

                # Check is out_var_str is defined, else create it
                if out_var_str is None:
                    out_var_str = in_var_str + "_dz"

                # Create new DataArray with the same dimensions as the parent
                # Crucially have a coordinate value that is appropriate to the target location.
                blank = xr.zeros_like(var.isel(z_dim=[0]))  # Using "z_dim=[0]" as a list preserves z-dimension
                blank.coords["depth_0"] -= blank.coords["depth_0"]  # reset coord vals to zero
                # Add blank slice to the 'surface'. Concat over the 'dim' coords
                diff = xr.concat([blank, var.diff(dim)], dim)
                diff_ndim, e3w_ndim = xr.broadcast(diff, out_obj.dataset.e3_0.squeeze())
                # Compute the derivative
                out_obj.dataset[out_var_str] = -diff_ndim / e3w_ndim

                # Assign attributes
                new_units = var.units + "/" + out_obj.dataset.depth_0.units
                # Convert to a xr.DataArray and return
                out_obj.dataset[out_var_str].attrs = {"units": new_units, "standard_name": out_var_str}

                # Return in object.
                return out_obj

            else:
                warn("Not ready for that combination of grid ({}) and " "derivative ({})".format(self.grid_ref, dim))
                return None
        else:
            warn(f"{in_var_str} does not exist in {get_slug(self)} dataset")
            return None

    def apply_doodson_x0_filter(self, var_str):
        """Applies Doodson X0 filter to a variable.

        Input variable is expected to be hourly.
        Output is saved back to original dataset as {var_str}_dxo

        !!WARNING: Will load in entire variable to memory. If dataset large,
        then subset before using this method or ensure you have enough free
        RAM to hold the variable (twice).

        DB:: Currently not tested in unit_test.py"""
        var = self.dataset[var_str]
        new_var_str = var_str + "_dx0"
        old_dims = var.dims
        time_index = old_dims.index("t_dim")
        filtered = stats_util.doodson_x0_filter(var, ax=time_index)
        if filtered is not None:
            self.dataset[new_var_str] = (old_dims, filtered)
        return

    @staticmethod
    def get_e3_from_ssh(nemo_t, e3t=True, e3u=False, e3v=False, e3f=False, e3w=False, dom_fn: str = None):
        """
        Where the model has been run with a nonlinear free surface
        and z* variable volumne (ln_vvl_zstar=True) then the vertical scale factors
        will vary in time (and space). This function will compute the vertical
        scale factors e3t, e3u, e3v, e3f and e3w by using the sea surface height
        field (ssh variable) and initial scale factors from the domain_cfg file.
        The vertical scale factors will be computed at the same model time as the
        ssh and if the ssh field is averaged in time then the scale factors will
        also be time averages.

        A t-grid NEMO object containing the ssh variable must be passed in. Either
        the domain_cfg path must have been passed in as an argument when the NEMO
        object was created or it must be passed in here using the dom_fn argument.

        e.g. e3t,e3v,e3f = coast.NEMO.get_e3_from_ssh(nemo_t,true,false,true,true,false)

        Parameters
        ----------
        nemo_t : (Coast.NEMO), NEMO object on the t-grid containing the ssh variable
        e3t : (boolean), true if e3t is to be returned. Default True.
        e3u : (boolean), true if e3u is to be returned. Default False.
        e3v : (boolean), true if e3v is to be returned. Default False.
        e3f : (boolean), true if e3f is to be returned. Default False.
        e3w : (boolean), true if e3w is to be returned. Default False.
        dom_fn : (str), Optional, path to domain_cfg file.

        Returns
        -------
        Tuple of xarray.DataArrays
        (e3t, e3u, e3v, e3f, e3w)
        Only those requested will be returned, but the ordering is always the same.

        """
        e3_return = []
        try:
            ssh = nemo_t.dataset.ssh
        except AttributeError:
            print("The nemo_t dataset must contain the ssh variable.")
            return
        if "t_dim" not in ssh.dims:
            ssh = ssh.expand_dims("t_dim", axis=0)

        # Load domain_cfg
        if dom_fn is None:
            dom_fn = nemo_t.filename_domain
        try:
            ds_dom = xr.open_dataset(dom_fn).squeeze().rename({"z": "z_dim", "x": "x_dim", "y": "y_dim"})
        except OSError:
            print(f"Problem opening domain_cfg file: {dom_fn}")
            return

        e3t_0 = ds_dom.e3t_0

        # Water column thickness, i.e. depth of bottom w-level on horizontal t-grid
        H = e3t_0.cumsum(dim="z_dim").isel(z_dim=ds_dom.bottom_level.astype("int") - 1)
        # Add correction to e3t_0 due to change in ssh
        e3t_new = e3t_0 * (1 + ssh / H)
        # preserve dimension ordering
        e3t_new = e3t_new.transpose("t_dim", "z_dim", "y_dim", "x_dim")
        # mask out correction at layers below bottom level
        e3t_new = e3t_new.where(e3t_new.z_dim < ds_dom.bottom_level, e3t_0.data)
        # preserve any other t mask
        e3t_new = e3t_new.where(~np.isnan(ssh))
        if e3t:
            e3_return.append(e3t_new.squeeze())

        if np.any([e3u, e3v, e3f]):
            e1e2t = ds_dom.e1t * ds_dom.e2t
        if np.any([e3u, e3v, e3w]):
            e3t_dt = e3t_new - e3t_0

        # area averaged interpolation onto the u-grid to get e3u
        if np.any([e3u, e3f]):
            e1e2u = ds_dom.e1u * ds_dom.e2u
            # interpolate onto u-grid
            e3u_temp = (
                (0.5 / e1e2u[:, :-1]) * ((e1e2t[:, :-1] * e3t_dt[:, :, :, :-1]) + (e1e2t[:, 1:] * e3t_dt[:, :, :, 1:]))
            ).transpose("t_dim", "z_dim", "y_dim", "x_dim")
            # u mask
            e3u_temp = e3u_temp.where(e3t_dt[:, :, :, 1:] != 0, 0)
            # mask out correction at layers below bottom level
            e3u_temp = e3u_temp.where(e3u_temp.z_dim < ds_dom.bottom_level[:, :-1], 0)
            # Add correction to e3u_0
            e3u_temp = e3u_temp + ds_dom.e3u_0[:, :, :-1]
            e3u_new = xr.zeros_like(e3t_new)
            e3u_new = e3u_new.load()
            e3u_new[:, :, :, :-1] = e3u_temp
            e3u_new[:, :, :, -1] = ds_dom.e3u_0[:, :, -1]
            e3u_new["longitude"] = ds_dom.glamu
            e3u_new["latitude"] = ds_dom.gphiu
            if e3u:
                e3_return.append(e3u_new.squeeze())

        # area averaged interpolation onto the u-grid to get e3v
        if e3v:
            e1e2v = ds_dom.e1v * ds_dom.e2v
            e3v_temp = (
                (0.5 / e1e2v[:-1, :]) * ((e1e2t[:-1, :] * e3t_dt[:, :, :-1, :]) + (e1e2t[1:, :] * e3t_dt[:, :, 1:, :]))
            ).transpose("t_dim", "z_dim", "y_dim", "x_dim")
            e3v_temp = e3v_temp.where(e3t_dt[:, :, 1:, :] != 0, 0)
            e3v_temp = e3v_temp.where(e3v_temp.z_dim < ds_dom.bottom_level[:-1, :], 0)
            e3v_temp = e3v_temp + ds_dom.e3v_0[:, :-1, :]
            e3v_new = xr.zeros_like(e3t_new)
            e3v_new = e3v_new.load()
            e3v_new[:, :, :-1, :] = e3v_temp
            e3v_new[:, :, -1, :] = ds_dom.e3v_0[:, -1, :]
            e3v_new["longitude"] = ds_dom.glamv
            e3v_new["latitude"] = ds_dom.gphiv
            e3_return.append(e3v_new.squeeze())

        # area averaged interpolation onto the u-grid to get e3f
        if e3f:
            e1e2f = ds_dom.e1f * ds_dom.e2f
            e3u_dt = e3u_new - ds_dom.e3u_0
            e3f_temp = (
                (0.5 / e1e2f[:-1, :]) * ((e1e2u[:-1, :] * e3u_dt[:, :, :-1, :]) + (e1e2u[1:, :] * e3u_dt[:, :, 1:, :]))
            ).transpose("t_dim", "z_dim", "y_dim", "x_dim")
            e3f_temp = e3f_temp.where(e3u_dt[:, :, 1:, :] != 0, 0)
            e3f_temp = e3f_temp.where(e3f_temp.z_dim < ds_dom.bottom_level[:-1, :], 0)
            e3f_temp = e3f_temp + ds_dom.e3f_0[:, :-1, :]
            e3f_new = xr.zeros_like(e3t_new)
            e3f_new = e3f_new.load()
            e3f_new[:, :, :-1, :] = e3f_temp
            e3f_new[:, :, -1, :] = ds_dom.e3f_0[:, -1, :]
            e3f_new["longitude"] = ds_dom.glamf
            e3f_new["latitude"] = ds_dom.gphif
            e3_return.append(e3f_new.squeeze())

        # simple vertical interpolation for e3w. Special treatment of top and bottom levels
        if e3w:
            # top levels correction same at e3t
            e3w_new = (ds_dom.e3w_0 + e3t_dt).transpose("t_dim", "z_dim", "y_dim", "x_dim")
            # levels between top and bottom
            e3w_new = e3w_new.load()
            e3w_new[dict(z_dim=slice(1, None))] = (
                0.5 * e3t_dt[:, :-1, :, :] + 0.5 * e3t_dt[:, 1:, :, :] + ds_dom.e3w_0[1:, :, :]
            )
            # bottom and below levels
            e3w_new = e3w_new.where(e3w_new.z_dim < ds_dom.bottom_level, e3t_dt.shift(z_dim=1) + ds_dom.e3w_0)
            e3_return.append(e3w_new.squeeze())

        return tuple(e3_return)

    def harmonics_combine(self, constituents, components=["x", "y"]):
        """
        Contains a new NEMO object containing combined harmonic information
        from the original object.

        NEMO saves harmonics to individual variables such as M2x, M2y... etc.
        This routine will combine these variables (depending on constituents)
        into a single data array. This new array will have the new dimension
        'constituent' and a new data coordinate 'constituent_name'.

        Parameters
        ----------
        constituents : List of strings containing constituent names to combine.
                       The case of these strings should match that used in
                       NEMO output. If a constituent is not found, no problem,
                       it just won't be in the combined dataset.
        components   : List of strings containing harmonic components to look
                       for. By default, this looks for the complex components
                       'x' and 'y'. E.g. if constituents = ['M2'] and
                       components is left as default, then the routine looks
                       for ['M2x', and 'M2y'].

        Returns
        -------
        NEMO() object, containing combined harmonic variables in a new dataset.
        """

        # Select only the specified constituents. NEMO model harmonics names are
        # things like "M2x" and "M2y". Ignore current harmonics. Start by constructing
        # the possible variable names
        names_x = np.array([cc + components[0] for cc in constituents])
        names_y = np.array([cc + components[1] for cc in constituents])
        constituents = np.array(constituents, dtype="str")

        # Compare against names in file
        var_keys = np.array(list(self.dataset.keys()))
        indices = [np.where(names_x == ss) for ss in names_x if ss in var_keys]
        indices = np.array(indices).T.squeeze()

        # Index the possible names to match file names
        names_x = names_x[indices]
        names_y = names_y[indices]
        constituents = constituents[indices]

        # Concatenate x and y variables into one array
        x_arrays = [self.dataset[ss] for ss in names_x]
        harmonic_x = "harmonic_" + components[0]
        x_data = xr.concat(x_arrays, dim="constituent").rename(harmonic_x)
        y_arrays = [self.dataset[ss] for ss in names_y]
        harmonic_y = "harmonic_" + components[1]
        y_data = xr.concat(y_arrays, dim="constituent").rename(harmonic_y)

        nemo_harmonics = Gridded()
        nemo_harmonics.dataset = xr.merge([x_data, y_data])
        nemo_harmonics.dataset["constituent"] = constituents

        return nemo_harmonics

    def harmonics_convert(
        self,
        direction="cart2polar",
        x_var="harmonic_x",
        y_var="harmonic_y",
        a_var="harmonic_a",
        g_var="harmonic_g",
        degrees=True,
    ):
        """
        Converts NEMO harmonics from cartesian to polar or vice versa.
        Make sure this NEMO object contains combined harmonic variables
        obtained using harmonics_combine().

        *Note:

        Parameters
        ----------
        direction (str) : Choose 'cart2polar' or 'polar2cart'. If 'cart2polar'
                          Then will look for variables x_var and y_var. If
                          polar2cart, will look for a_var (amplitude) and
                          g_var (phase).
        x_var (str)     : Harmonic x variable name in dataset (or output)
                          default = 'harmonic_x'.
        y_var (str)     : Harmonic y variable name in dataset (or output)
                          default = 'harmonic_y'.
        a_var (str)     : Harmonic amplitude variable name in dataset (or output)
                          default = 'harmonic_a'.
        g_var (str)     : Harmonic phase variable name in dataset (or output)
                          default = 'harmonic_g'.
        degrees (bool)  : Whether input/output phase are/will be in degrees.
                          Default is True.

        Returns
        -------
        Modifies NEMO() dataset in place. New variables added.
        """
        if direction == "cart2polar":
            a, g = general_utils.cartesian_to_polar(self.dataset[x_var], self.dataset[y_var], degrees=degrees)
            self.dataset[a_var] = a
            self.dataset[g_var] = g
        elif direction == "polar2cart":
            x, y = general_utils.polar_to_cartesian(self.dataset[a_var], self.dataset[g_var], degrees=degrees)
            self.dataset[x_var] = x
            self.dataset[y_var] = y
        else:
            print("Unknown direction setting. Choose cart2polar or polar2cart")

        return

    def time_slice(self, date0, date1):
        """Return new Gridded object, indexed between dates date0 and date1"""
        dataset = self.dataset
        t_ind = pd.to_datetime(dataset.time.values) >= date0
        dataset = dataset.isel(t_dim=t_ind)
        t_ind = pd.to_datetime(dataset.time.values) < date1
        dataset = dataset.isel(t_dim=t_ind)
        gridded_out = Gridded()
        gridded_out.dataset = dataset
        return gridded_out
