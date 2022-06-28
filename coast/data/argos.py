"""Argos class"""
from .index import Indexed
import numpy as np
import xarray as xr
import pandas as pd
from .._utils.logging_util import get_slug, debug
from typing import Union
from pathlib import Path


class Argos(Indexed):
    """Class for reading Argos CSV formatted data files into an xarray object"""

    def __init__(self, file_path: str = None, config: Union[Path, str] = None):
        """Init Argos data object

        Args:
            file_path (str) : path of data fil
            config (Path or str) : configuration file
        """
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(config)
        self.read_data(file_path)
        self.apply_config_mappings()
        print(f"{get_slug(self)} initialised")

    def read_data(self, file_path: str) -> None:
        """Read the data file

        Expected format and variable names are

            DATIM,LAT,LON,SST,SST_F,PSST,PSST_F,PPRES,PPRES_F,BEN

        xarray dataset to have dimension as time and coordinates as time, latitude and longitude

        Args:
            file_path (str) : path of data file
        """
        df = pd.read_csv(file_path, na_values=" ")

        self.dataset = xr.Dataset()
        var_names = list(df.columns)
        for name in var_names:
            if name == "DATIM":
                time = np.array(pd.to_datetime(df[name]))
                self.dataset[name] = xr.DataArray(list(time), dims=["time"])
                self.dataset = self.dataset.set_coords(name)
            else:
                self.dataset[name] = xr.DataArray(list(np.array(df[name])), dims=["time"])
