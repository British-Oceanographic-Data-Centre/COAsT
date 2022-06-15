"""Oceanparcels class for reading ocean parcels data."""
from .lagrangian import Lagrangian
import xarray as xr
from .._utils.logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path


class Oceanparcels(Lagrangian):
    """Reading ocean parcels data (netcdf format) into an xarray object."""

    def __init__(self, file_path: str = None, config: Union[Path, str] = None):
        """Initialization and file reading.

        Args:
           file_path (str): path to data file
           config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(config)

        if file_path is None:
            warn("Object created but no file or directory specified: \n" "{0}".format(str(self)), UserWarning)
        else:
            self.load_single(file_path)
            self.apply_config_mappings()

        print(f"{get_slug(self)} initialised")

    def load_single(self, file_path) -> None:
        """Loads a single file into object's dataset variable.

        Args:
            file_path (str): path to data file
        """
        self.dataset = xr.open_dataset(file_path, chunks=self.chunks)
