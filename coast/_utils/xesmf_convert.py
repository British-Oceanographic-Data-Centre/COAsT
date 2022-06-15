"""A class to convert from coast gridded to xesmf."""
import os.path as path_lib
import warnings
from ..data.gridded import Gridded

# from dask import delayed, compute, visualize
# import graphviz
import numpy as np
import xarray as xr


class xesmf_convert:
    """
    Converts the main dataset within a COAsT.Gridded object to be suitable
    for input to XESMF for regridding to either a curvilinear or rectilienar
    grid. All you need to do if provide a Gridded object and a grid type when
    creating a new instance of this class. It will then contain an appropriate
    input dataset. You may also provide a second COAsT gridded object if
    regridding between two objects. For using xesmf, please see the package's
    documentation website here:

    https://xesmf.readthedocs.io/en/latest/index.html

    You can install XESMF using:

        conda install -c conda-forge xesmf.

    The setup used by this class has been tested for xesmf v0.6.2 alongside
    esmpy v8.0.0. It was installed using:

        conda install -c conda-forge xesmf esmpy=8.0.0

    INPUTS
     input_gridded_obj (Gridded) :: Gridded object to be regridded
     output_gridded_obj(Gridded) :: (optional) Gridded object to regrid TO
     reorder_dims      (bool)    :: Xesmf requires that lat/lon dimensions are
                                    the last dimensions. If this is True, then
                                    will attempt to reorder dimensions. Not
                                    recommended for large datasets.
                                    [Default = False]

    >>> EXAMPLE USEAGE <<<
    If regridding a Gridded object to an arbitrarily defined rectilinear
    or curvilinear grid, you just need to do the following:

        import xesmf as xe

        # Create your gridded object
        gridded = coast.Gridded(*args, **kwargs)

        # Pass the gridded object over to xesmf_convert
        xesmf_ready = coast.xesmf_convert(gridded, input_grid_type = 'curvilinear')

        # Now this object will contain a dataset called xesmf_input, which can
        # be passed over to xesmf. E.G:

        destination_grid = xesmf.util.grid_2d(-15, 15, 1, 45, 65, 1)
        regridder = xe.Regridder(xesmf_ready.input_grid, destination_grid,
                                 "bilinear")
        regridded_dataset = regridder(xesmf_ready.input_data)

    XESMF contains a couple of difference functions for quickly creating output
    grids, such as xesmf.util.grid_2d and xesmf.util.grid_global(). See their
    website for more info.

    The process is almost the same if regridding from one COAsT.Gridded object
    to another (gridded0 -> gridded1):

        xesmf_ready = coast.xesmf_convert(gridded0, gridded1,
                                          input_grid_type = "curvilinear",
                                          output_grid_type = "curvilinear")
        regridder = xe.Regridder(xesmf_ready.input_grid,
                                 xesmf_ready.output_grid, "bilinear")
        regridded_dataset = regridder(xesmf_ready.input_data)

    Note that you can select which variables you want to regrid, either prior
    to using this tool or by indexing the input_data dataset. e.g.:

        regridded_dataset = regridder(xesmf_ready.input_data['temperature'])

    If your input datasets were lazy loaded, then so will the regridded dataset.
    At this point you can either load the data or (recomended) save the regridded
    data to file:

        regridded_dataset.to_netcdf(<filename_to_save>)

    Before saving back to file, call xesmf_ready.to_gridded() to convert
    the regridded xesmf object back to a gridded object

    """

    def __init__(
        self,
        input_gridded_obj=None,
        output_gridded_obj=None,
        input_grid_type="curvilinear",
        output_grid_type="curvilinear",
        reorder_dims=False,
    ):
        """
        Handles the inputs and reformatting of data. See Class docstring for
        more info.
        """

        # Set grid types
        self.input_grid_type = input_grid_type
        self.output_grid_type = output_grid_type

        # If an input object is provided, convert to grid/data for xesmf
        if input_gridded_obj is not None:
            input_dataset = input_gridded_obj.dataset
            self.input_grid, self.input_data = self._get_xesmf_datasets(input_dataset, input_grid_type)

        # If an output object is provided, convert to grid/data for xesmf
        if output_gridded_obj is not None:
            output_dataset = output_gridded_obj.dataset
            self.output_grid, _ = self._get_xesmf_datasets(output_dataset, output_grid_type)

    @classmethod
    def _get_xesmf_datasets(cls, dataset, grid_type, reorder_dims=False):
        """For a given dataset taken from a Gridded object and a grid_type
        (curvilinear or rectilinear), determine the xesmf formatted dataset.
        This method does some checks to make sure the dataset is suitable and
        renames the relevant dimensions/coordinates. Any vars that don't have
        both x_dim and y_dim will be dropped. If x_dim and y_dim are present
        BUT they are not the last dimensions AND reorder_dims=True then
        the dimensions will be reordered (not good for lazy loading/chunking).
        """

        # Check that x_dim and y_dim are in the dataset. If not, raise error
        if "x_dim" not in dataset.dims or "y_dim" not in dataset.dims:
            raise AttributeError("Gridded.dataset must  contain two spatial dimensions called x_dim and y_dim")

        # Check each variable has x_dim AND y_dim and that they are the
        # last dimensions. If not, then drop the variables if reorder_dims=False.
        # Otherwise, attempt to reorder dimensions.

        # Rename Dimensions and coordinates
        if grid_type == "curvilinear":
            dataset = dataset.swap_dims({"x_dim": "x", "y_dim": "y"})
            dataset = dataset.rename_vars({"latitude": "lat", "longitude": "lon"})

        # Define grid and data
        grid = dataset[["lon", "lat"]]
        data = dataset
        return grid, data

    @staticmethod
    def to_gridded(xesmf_dataset):
        """Converts an xesmf_dataset back to a Coast.Gridded() object. Returns
        a Gridded object."""

        # Create empty gridded object
        gridded_obj = Gridded()

        # Rename dimensions and coordinates
        xesmf_dataset = xesmf_dataset.swap_dims({"x": "x_dim", "y": "y_dim"})
        xesmf_dataset = xesmf_dataset.rename_vars({"lat": "latitude", "lon": "longitude"})

        # Place renamed dataset into new gridded object.
        gridded_obj.dataset = xesmf_dataset

        return gridded_obj
