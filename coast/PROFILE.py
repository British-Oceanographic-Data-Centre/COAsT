from .INDEX import INDEXED
import numpy as np
import xarray as xr
from . import general_utils, plot_util, crps_util, COAsT
import matplotlib.pyplot as plt
import glob
import datetime
from .logging_util import get_slug, debug, info, warn, warning


class PROFILE(INDEXED):
    """
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimensions:

        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a
                     individual location.
        > level   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files.
    """

    def __init__(self, file=None, chunks={}, multiple=False):
        """ Initialization and file reading."""
        self.dataset = None
        self.set_dimension_mapping()
        self.set_variable_mapping()

        if file is None:
            warn(
                "Object created but no file or directory specified: \n"
                "{0}".format(str(self)),
                UserWarning
            )
        else:
            self.read_EN4(file, chunks, multiple)
            self.apply_var_and_dim_mappings_to_dataset()

        print(f"{get_slug(self)} initialised")

    def read_EN4(self, fn_en4, chunks={}, multiple=False):
        """Reads a single or multiple EN4 netCDF files into the COAsT profile data structure."""

        print("PROFILE read_EN4")

        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks=chunks)
        else:
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]

            file_to_read = []
            for file in fn_en4:
                if '*' in file:
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
                    self.dataset = xr.concat((self.dataset, data_tmp), dim='N_PROF')

    def apply_var_and_dim_mappings_to_dataset(self):
        # rename dimensions and variables (keeps only those variables that are in the mappings)
        self.dataset = self.dataset.set_coords(['LATITUDE', 'LONGITUDE', 'JULD'])

        if self.dim_mapping is not None:
            self.dataset = self.dataset.rename(self.dim_mapping)

        if self.var_mapping is not None:
            vars_to_keep = list(self.var_mapping.keys())
            self.dataset = self.dataset[vars_to_keep].rename_vars(self.var_mapping)

    def set_dimension_mapping(self):
        self.dim_mapping = {'N_PROF': 'profile',
                            'N_PARAM': 'parameter',
                            'N_LEVELS': 'z_dim'
                            }
        print(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")

    def set_variable_mapping(self):
        self.var_mapping = {'LATITUDE': 'latitude',
                            'LONGITUDE': 'longitude',
                            'DEPH_CORRECTED': 'depth',
                            'JULD': 'time',
                            'POTM_CORRECTED': 'potential_temperature',
                            'TEMP': 'temperature',
                            'PSAL_CORRECTED': 'practical_salinity',
                            'POTM_CORRECTED_QC': 'qc_potential_temperature',
                            'PSAL_CORRECTED_QC': 'qc_practical_salinity',
                            'DEPH_CORRECTED_QC': 'qc_depth',
                            'JULD_QC': 'qc_time'
                            }
        print(f"{get_slug(self)} var_mapping set to {self.var_mapping}")

    def __getitem__(self, name: str):
        return self.dataset[name]

    """======================= Manipulate ======================="""

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.
        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        ind = general_utils.subset_indices_lonlat_box(self.dataset.longitude,
                                                      self.dataset.latitude,
                                                      lonbounds[0], lonbounds[1],
                                                      latbounds[0], latbounds[1])
        return ind

    """======================= Plotting ======================="""

    def plot_profile(self, var: str, profile_indices=None):

        fig = plt.figure(figsize=(7, 10))

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims['profile'])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(profile=ii)
            prof_depth = self.dataset.depth.isel(profile=ii)
            ax = plt.plot(prof_var, prof_depth)

        plt.gca().invert_yaxis()
        plt.xlabel(var + '(' + self.dataset[var].units + ')')
        plt.ylabel('Depth (' + self.dataset.depth.units + ')')
        plt.grid()
        return fig, ax

    def plot_map(self, profile_indices=None, var_str=None, depth_index=None):

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims['profile'])

        profiles = self.dataset.isel(profile=profile_indices)

        if var_str is None:
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values, s=5)
        else:
            print(profiles)
            c = profiles[var_str].isel(level=depth_index)
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values,
                                            c=c, s=5)
        return fig, ax

    def plot_ts_diagram(self, profile_index, var_t='potential_temperature',
                        var_s='practical_salinity'):

        profile = self.dataset.isel(profile=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)

        return fig, ax

    """======================= Model Comparison ======================="""
