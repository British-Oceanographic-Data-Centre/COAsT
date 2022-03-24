"""Profile_WOD Class"""
from .index import Indexed
import numpy as np
import xarray as xr
from . import general_utils, plot_util
import matplotlib.pyplot as plt
import glob
import datetime
from .logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
import xarray.ufuncs as uf
import pandas as pd


class Profile_WOD(Indexed):
    """
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the WOD and so it contains 1D profiles (profile * depth levels) that we need
    to restructure into a 2D array. Note that its variable has its own dimention

        > casts ::   contains the individual locations of observations
        > z_N   ::   The dimension for depth levels (in this case in a 
                     common depth as (regrided by NOAA)
       > X_N    ::   contains the individual locations of observations
                     as profile * depth levels, which varies for each
                     variable (e.g., Temperature_N, Salinity_N, Oxygen_N ...)
    """

    def __init__(self, file_path: str = None, multiple=False, config: Union[Path, str] = None):
        """Initialization and file reading.

        Args:
            file_path (str): path to data file
            multiple (boolean): True if reading multiple files otherwise False
            config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(config)

        if file_path is None:
            warn("Object created but no file or directory specified: \n" "{0}".format(str(self)), UserWarning)
        else:
            self.read_en4(file_path, self.chunks, multiple)
            self.apply_config_mappings()

        debug(f"{get_slug(self)} initialised")

    def read_en4(self, fn_en4, chunks: dict = {}, multiple=False) -> None:
        """Reads a single or multiple EN4 netCDF files into the COAsT profile data structure.

        Args:
            fn_en4 (str): path to data file
            chunks (dict): chunks
            multiple (boolean): True if reading multiple files otherwise False
        """
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks=chunks)
        else:
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]

            file_to_read = []
            for file in fn_en4:
                if "*" in file:
                    wildcard_list = glob.glob(file)
                    file_to_read = file_to_read + wildcard_list
                else:
                    file_to_read.append(file)

            # Reorder files to read
            file_to_read = np.array(file_to_read)
            dates = [ff[-9:-3] for ff in file_to_read]
            dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]), 1) for dd in dates]
            sort_ind = np.argsort(dates)
            file_to_read = file_to_read[sort_ind]

            for ff in range(0, len(file_to_read)):
                file = file_to_read[ff]
                data_tmp = xr.open_dataset(file, chunks=chunks)
                if ff == 0:
                    self.dataset = data_tmp
                else:
                    self.dataset = xr.concat((self.dataset, data_tmp), dim="N_PROF")

#    """================Regrid to 2D================"""
#    def transform_2D(self, :
#        """transforms the 1D variable into 2D variable with dimensions of:
#           (profile, N_depth)
#        Args:
#            X          -- the variable (e.g., Salinity, DIC etc.)
#            X_N     -- the dimensions of each variable observations
#            profile -- casts (location of observations cast, 
#                       common in all variables, and there will be nan if obsetvations
#                       not available for a variable)
#            X_row_size -- give the vertical index (number of depths)
#                          for each cast and each variable 
#        """



