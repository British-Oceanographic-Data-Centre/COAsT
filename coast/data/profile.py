"""Profile Class"""
from .index import Indexed
import numpy as np
import xarray as xr
import gsw
from .._utils import general_utils, plot_util
import matplotlib.pyplot as plt
import glob
import datetime
from .._utils.logging_util import get_slug, debug, info, warn, warning, error

from typing import Union
from pathlib import Path
import pandas as pd


class Profile(Indexed):
    """
    INDEXED type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based around having
    discrete profile locations with independent depth dimensions and coords.
    The class dataset should contain two dimensions:

        > id_dim      :: The profiles dimension. Each element of this dimension
                     contains data (e.g. cast) for an individual location.
        > z_dim   :: The dimension for depth levels. A profile object does not
                     need to have shared depths, so NaNs might be used to
                     pad any depth array.

    Alongside these dimensions, the following minimal coordinates should also
    be available:

        > longitude (id_dim)   :: 1D array of longitudes, one for each id_dim
        > latitude  (id_dim)   :: 1D array of latitudes, one for each id_dim
        > time      (id_dim)   :: 1D array of times, one for each id_dim
        > depth     (id_dim, z_dim)  :: 2D array of depths, with different depth
                                    levels being provided for each profile.
                                    Note that these depth levels need to be
                                    stored in a 2D array, so NaNs can be used
                                    to pad out profiles with shallower depths.
        > id_name   (id_dim)   :: [Optional] Name of id_dim/case or id_dim number.

    You may create an empty profile object by using profile = coast.Profile().
    You may then add your own dataset to the object profile or use one of the
    functions within Profile() for reading common profile datasets:

        > read_en4()
        > read_wod()

    Optionally, you may pass a dataset to the Profile object on creation:

        profile = coast.Profile(dataset = profile_dataset)

    A config file can also be provided, in which case any netcdf read functions
    will rename dimensions and variables as dictated.
    """

    def __init__(self, dataset=None, config: Union[Path, str] = None):
        """Initialization and file reading. You may initialize

        Args:
            config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")
        self.config = config
        super().__init__(self.config)

        # If dataset is provided, put inside this object
        if dataset is not None:
            self.dataset = dataset
            self.apply_config_mappings()

        debug(f"{get_slug(self)} initialised")

    def read_en4(self, fn_en4, chunks: dict = {}, multiple=False) -> None:
        """
        Reads a single or multiple EN4 netCDF files into the COAsT profile
        data structure.

        Parameters
        ----------
        fn_en4 : TYPE
            path to data file.
        chunks : dict, optional
            Chunking specification
        multiple : TYPE, optional
            True if reading multiple files otherwise False

        Returns
        -------
        None. Populates dataset within Profile object.
        """

        # If not multiple then just read the netcdf file
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks=chunks)

        # If multiple, then we have to get all file names and read them in a
        # loop, followed by concatenation
        else:
            # Check a list is provided
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]

            # Use glob to get a list of file paths from input
            file_to_read = []
            for file in fn_en4:
                if "*" in file:
                    wildcard_list = glob.glob(file)
                    file_to_read = file_to_read + wildcard_list
                else:
                    file_to_read.append(file)

            # Reorder files to read
            file_to_read = np.array(file_to_read)
            dates = [ff[-9:-3] for ff in file_to_read]  # Assumes monthly filename structure: EN*yyyymm.nc
            dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]), 1) for dd in dates]
            sort_ind = np.argsort(dates)
            file_to_read = file_to_read[sort_ind]

            # Loop over all files, open them and concatenation them into one
            for ff in range(0, len(file_to_read)):
                file = file_to_read[ff]
                data_tmp = xr.open_dataset(file, chunks=chunks)
                if ff == 0:
                    self.dataset = data_tmp
                else:
                    self.dataset = xr.concat((self.dataset, data_tmp), dim="N_PROF")

        # Apply config settings
        self.apply_config_mappings()

    def read_wod(self, fn_wod, chunks: dict = {}) -> None:
        """Reads a single World Ocean Database netCDF files into the COAsT profile data structure.

        Args:
            fn_wod (str): path to data file
            chunks (dict): chunks
        """
        self.dataset = xr.open_dataset(fn_wod, chunks=chunks)
        self.apply_config_mappings()

    """======================= Manipulate ======================="""

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Get a subset of this Profile() object in a spatial box.

        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]

        return: A new profile object containing subsetted data
        """
        ind = general_utils.subset_indices_lonlat_box(
            self.dataset.longitude, self.dataset.latitude, lonbounds[0], lonbounds[1], latbounds[0], latbounds[1]
        )
        return Profile(dataset=self.dataset.isel(id_dim=ind[0]))

    """======================= Plotting ======================="""

    def plot_profile(self, var: str, profile_indices=None):
        fig = plt.figure(figsize=(7, 10))

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims["id_dim"])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(id_dim=ii)
            prof_depth = self.dataset.depth.isel(id_dim=ii)
            ax = plt.plot(prof_var, prof_depth)

        plt.gca().invert_yaxis()
        plt.xlabel(var + "(" + self.dataset[var].units + ")")
        plt.ylabel("Depth (" + self.dataset.depth.units + ")")
        plt.grid()
        return fig, ax

    def plot_map(self, var_str=None):
        profiles = self.dataset

        if var_str is None:
            fig, ax = plot_util.geo_scatter(profiles.longitude.values, profiles.latitude.values, s=5)
        else:
            c = profiles[var_str]
            fig, ax = plot_util.geo_scatter(profiles.longitude.values, profiles.latitude.values, c=c, s=5)
        return fig, ax

    def plot_ts_diagram(self, profile_index, var_t="potential_temperature", var_s="practical_salinity"):
        profile = self.dataset.isel(id_dim=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)

        return fig, ax

    """======================= Model Comparison ======================="""

    def process_en4(self, sort_time=True, remove_flagged_neighbours=False):
        """
        VERSION 1.4 (05/07/2021)

        PREPROCESSES EN4 data ready for comparison with model data.
        This routine will cut out a desired geographical box of EN4 data and
        then apply quality control according to the available flags in the
        netCDF files. Quality control happens in two steps:
            1. Where a whole data profile is flagged, it is completely removed
               from the dataset
            2. Where a single datapoint is rejected in either temperature or
               salinity, it is set to NaN.
        This routine attempts to use xarray/dask chunking magic to keep
        memory useage low however some memory is still needed for loading
        flags etc. May be slow if using large EN4 datasets.

        Routine will return a processed profile object dataset and can write
        the new dataset to file if fn_out is defined. If saving to the
        PROFILE object, be aware that DASK computations will not have happened
        and will need to be done using .load(), .compute() or similar before
        accessing the values. IF using multiple EN4 files or large dataset,
        make sure you have chunked the data over N_PROF dimension.

        INPUTS
         fn_out (str)      : Full path to a desired output file. If unspecified
                             then nothing is written.

        remove_flagged_neighbours: EN offers a profile flag that indicates there are
                other profiles within 0.2 deg of latitude and longitude and 1 hour that
                appear to be of higher quality. In previous versions of the dataset these
                profiles would have not been stored in the data files. Setting this flag
                as True removes these profiles.

        EXAMPLE USEAGE:
         profile = coast.PROFILE()
         profile.read_EN4(fn_en4, chunks={'N_PROF':10000})
         fn_out = '~/output_file.nc'
         new_profile = profile.preprocess_en4(fn_out = fn_out,
                                              lonbounds = [-10, 10],
                                              latbounds = [45, 65])
        """

        ds = self.dataset

        # Load in the quality control flags
        debug(f" Applying QUALITY CONTROL to EN4 data...")
        ds.qc_flags_profiles.load()

        # This line reads converts the QC integer to a binary string.
        # Each bit of this string is a different QC flag. Which flag is which can
        # be found on the EN4 website:
        # https://www.metoffice.gov.uk/hadobs/en4/en4-0-2-profile-file-format.html
        qc_str = [
            np.binary_repr(ds.qc_flags_profiles.astype(int).values[pp]).zfill(32)[::-1]
            for pp in range(ds.sizes["id_dim"])
        ]

        # Determine indices of the profiles that we want to keep
        reject_tem_prof = np.array([int(qq[0]) for qq in qc_str], dtype=bool)
        reject_sal_prof = np.array([int(qq[1]) for qq in qc_str], dtype=bool)
        reject_both_prof = np.logical_and(reject_tem_prof, reject_sal_prof)
        if remove_flagged_neighbours:
            reject_close_flagged_prof = np.array([int(qq[2]) for qq in qc_str], dtype=bool)
            reject_both_prof = np.logical_or(reject_both_prof, reject_close_flagged_prof)
        ds["reject_tem_prof"] = (["id_dim"], reject_tem_prof)
        ds["reject_sal_prof"] = (["id_dim"], reject_sal_prof)
        debug("     >>> QC: Completely rejecting {0} / {1} id_dims".format(np.sum(reject_both_prof), ds.dims["id_dim"]))

        # Subset profile dataset to remove profiles that are COMPLETELY empty
        ds = ds.isel(id_dim=~reject_both_prof)
        reject_tem_prof = reject_tem_prof[~reject_both_prof]
        reject_sal_prof = reject_sal_prof[~reject_both_prof]
        debug(f" QC: Additional profiles converted to NaNs: ")
        debug(f"     >>> {0} temperature profiles ".format(np.sum(reject_tem_prof)))
        debug(f"     >>> {0} salinity profiles ".format(np.sum(reject_sal_prof)))

        debug(f"MASKING rejected profiles, replacing with NaNs...")
        reject_tem_prof_arr = ds.reject_tem_prof.broadcast_like(ds.potential_temperature).values
        reject_sal_prof_arr = ds.reject_sal_prof.broadcast_like(ds.practical_salinity).values
        ds["temperature"] = xr.where(~reject_tem_prof_arr, ds["temperature"], np.nan)
        ds["potential_temperature"] = xr.where(~reject_tem_prof_arr, ds["potential_temperature"], np.nan)
        ds["practical_salinity"] = xr.where(~reject_sal_prof_arr, ds["practical_salinity"], np.nan)

        # Get new QC flags array
        qc_lev = ds.qc_flags_levels.values
        #
        reject_tem_lev = np.zeros((ds.dims["id_dim"], ds.dims["z_dim"]), dtype=bool)
        reject_sal_lev = np.zeros((ds.dims["id_dim"], ds.dims["z_dim"]), dtype=bool)

        # Get lists of QC integers that marks levels for masking
        int_tem, int_sal, int_both = self.calculate_en4_qc_flags_levels()
        for ii in range(len(int_tem)):
            reject_tem_lev[qc_lev == int_tem[ii]] = 1
        for ii in range(len(int_sal)):
            reject_sal_lev[qc_lev == int_sal[ii]] = 1
        for ii in range(len(int_both)):
            reject_tem_lev[qc_lev == int_both[ii]] = 1
            reject_sal_lev[qc_lev == int_both[ii]] = 1

        ds["reject_tem_datapoint"] = (["id_dim", "z_dim"], reject_tem_lev)
        ds["reject_sal_datapoint"] = (["id_dim", "z_dim"], reject_sal_lev)

        debug(f"MASKING rejected datapoints, replacing with NaNs...")
        ds["temperature"] = xr.where(~reject_tem_lev, ds["temperature"], np.nan)
        ds["potential_temperature"] = xr.where(~reject_tem_lev, ds["potential_temperature"], np.nan)
        ds["practical_salinity"] = xr.where(~reject_sal_lev, ds["practical_salinity"], np.nan)

        if sort_time:
            debug(f"Sorting Time Dimension...")
            ds = ds.sortby("time")

        debug(f"Finished processing data. Returning new Profile object.")

        return_prof = Profile()
        return_prof.dataset = ds
        return return_prof

    def obs_operator(self, gridded, mask_bottom_level=True):
        """
        VERSION 2.0 (04/10/2021)
        Author: David Byrne

        Does a spatial and time interpolation of a gridded object's data.
        A nearest neighbour approach is used for both interpolations. Both
        datasets (the Profile and Gridded objects) must contain longitude,
        latitude and time coordinates. This routine expects there to be a
        landmask variable in the gridded object. This is is not available,
        then place an array of zeros into the dataset, with dimensions
        (y_dim, x_dim).

        This routine will do the interpolation based on the chunking applied
        to the Gridded object. Please ensure you have the available memory to
        have an entire Gridded chunk loaded to memory. If multiple files are
        used, then using one chunk per file will be most efficient. Time
        chunking is generally the better option for this routine.

        INPUTS:
         gridded (Gridded)        : gridded object created on t-grid
         mask_bottom_level (bool) : Whether or not to mask any data below the
                                    model's bottom level. If True, then ensure
                                    the Gridded object's dataset contain's a
                                    bottom_level variable with dims
                                    (y_dim, x_dim).

        OUTPUTS:
         Returns a new PROFILE object containing a computed dataset of extracted
         profiles.
        """

        # Read EN4, then extract desired variables
        en4 = self.dataset
        gridded = gridded.dataset

        # CHECKS
        # 1. Check that bottom_level is in dataset if mask_bottom_level is True
        if mask_bottom_level:
            if "bottom_level" not in gridded.variables:
                raise ValueError(
                    "bottom_level not found in input dataset. Please ensure variable is present or set mask_bottom_level to False"
                )

        # Use only observations that are within model time window.
        en4_time = en4.time.values
        mod_time = gridded.time.values

        # SPATIAL indices - nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2d(
            gridded["longitude"], gridded["latitude"], en4["longitude"], en4["latitude"], mask=gridded.landmask
        )
        debug(f"Spatial Indices Calculated")

        # TIME indices - model nearest to obs time
        en4_time = en4.time.values
        ind_t = [np.argmin(np.abs(mod_time - en4_time[tt])) for tt in range(en4.dims["id_dim"])]
        ind_t = xr.DataArray(ind_t)
        debug(f"Time Indices Calculated")

        # Find out which variables have both depth and profile
        # This is for applying the bottom_level mask later
        var_list = list(gridded.keys())
        bl_var_list = []
        for vv in var_list:
            cond1 = "z_dim" not in gridded[vv].dims
            if cond1:
                bl_var_list.append(vv)

        # Get chunks along the time dimension and determine whether chunks
        # are described by a single equal size, or a tuples of sizes
        time_chunks = gridded.unify_chunks().chunks["t_dim"]
        time_dim = gridded.dims["t_dim"]
        start_ii = 0  # Starting index for loading data. Increments each loop
        count_ii = 0  # Counting index for allocating data. Increments 1 each loop

        while start_ii < time_dim:
            end_ii = start_ii + time_chunks[count_ii]
            debug(f"{0}: {1} > {2}".format(count_ii, start_ii, end_ii))

            # Determine which time indices lie in this chunk
            ind_in_chunk = np.logical_and(ind_t >= start_ii, ind_t < end_ii)

            # Check There are some indices at all
            if np.sum(ind_in_chunk) == 0:
                start_ii = end_ii
                if count_ii == 0:
                    mod_profiles = xr.Dataset()

                count_ii = count_ii + 1
                continue

            # Pull out x,y and t indices
            ind_x_in_chunk = ind_x[ind_in_chunk]
            ind_y_in_chunk = ind_y[ind_in_chunk]
            ind_t_in_chunk = ind_t[ind_in_chunk] - start_ii

            # Index a temporary chunk and read it to memory
            ds_tmp = gridded.isel(t_dim=np.arange(start_ii, end_ii)).load()

            # Index loaded chunk and rename dim_0 to profile
            ds_tmp_indexed = ds_tmp.isel(x_dim=ind_x_in_chunk, y_dim=ind_y_in_chunk, t_dim=ind_t_in_chunk)
            ds_tmp_indexed = ds_tmp_indexed.rename({"dim_0": "id_dim"})

            # Mask out all levels deeper than bottom_level
            # Here I have used set_coords() and reset_coords() to omit variables
            # with no z_dim from the masking. Otherwise xr.where expands these
            # dimensions into full 2D arrays.
            if mask_bottom_level:
                n_z_tmp = ds_tmp_indexed.dims["z_dim"]
                bl_array = ds_tmp_indexed.bottom_level.values
                z_index, bl_index = np.meshgrid(np.arange(0, n_z_tmp), bl_array)
                mask2 = xr.DataArray(z_index < bl_index, dims=["id_dim", "z_dim"])
                ds_tmp_indexed = ds_tmp_indexed.set_coords(bl_var_list)
                ds_tmp_indexed = ds_tmp_indexed.where(mask2)
                ds_tmp_indexed = ds_tmp_indexed.reset_coords(bl_var_list)

            # If not first iteration, concatenate this indexed chunk onto
            # final output dataset
            if count_ii == 0:
                mod_profiles = ds_tmp_indexed
            else:
                if len(mod_profiles.data_vars) == 0:
                    mod_profiles = xr.merge((mod_profiles, ds_tmp_indexed))
                else:
                    mod_profiles = xr.concat((mod_profiles, ds_tmp_indexed), dim="id_dim")

            # Update counters
            start_ii = end_ii
            count_ii = count_ii + 1

        # Put obs time into the output array
        mod_profiles["obs_time"] = (["id_dim"], en4_time)

        # Calculate interpolation distances
        interp_dist = general_utils.calculate_haversine_distance(
            en4.longitude, en4.latitude, mod_profiles.longitude, mod_profiles.latitude
        )
        mod_profiles["interp_dist"] = (["id_dim"], interp_dist.values)

        # Calculate interpolation time lags
        interp_lag = (mod_profiles.time.values - en4_time).astype("timedelta64[h]")
        mod_profiles["interp_lag"] = (["id_dim"], interp_lag)

        # Put x and y indices into dataset
        mod_profiles["nearest_index_x"] = (["id_dim"], ind_x.values)
        mod_profiles["nearest_index_y"] = (["id_dim"], ind_y.values)
        mod_profiles["nearest_index_t"] = (["id_dim"], ind_t.values)
        return Profile(dataset=mod_profiles)

    def match_to_grid(self, gridded, limits=[0, 0, 0, 0], rmax=7000.0) -> None:
        """Match profiles locations to grid, finding 4 nearest neighbours for each profile.

        Args:
            gridded (Gridded): Gridded object.
            limits (List): [jmin,jmax,imin,imax] - Subset to this region.
            rmax (int): 7000 m - maxmimum search distance (metres).

        ### NEED TO DESCRIBE THE OUTPUT. WHAT DO i_prf, j_prf, rmin_prf REPRESENT?

        ### THIS LOOKS LIKE SOMETHING THE profile.obs_operator WOULD DO
        """

        if sum(limits) != 0:
            gridded.subset(ydim=range(limits[0], limits[1] + 0), xdim=range(limits[2], limits[3] + 1))
        # keep the grid or subset on the hydrographic profiles object
        gridded.dataset["limits"] = limits

        prf = self.dataset
        grd = gridded.dataset
        grd["landmask"] = grd.bottom_level == 0
        lon_prf = prf["longitude"]
        lat_prf = prf["latitude"]
        lon_grd = grd["longitude"]
        lat_grd = grd["latitude"]
        # SPATIAL indices - 4 nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2d(
            lon_grd, lat_grd, lon_prf, lat_prf, mask=grd.landmask, number_of_neighbors=4
        )
        ind_x = ind_x.values
        ind_y = ind_y.values

        # Exclude out of bound points
        i_exc = np.concatenate(
            (
                np.where(lon_prf < lon_grd.values.ravel().min())[0],
                np.where(lon_prf > lon_grd.values.ravel().max())[0],
                np.where(lat_prf < lat_grd.values.ravel().min())[0],
                np.where(lat_prf > lat_grd.values.ravel().max())[0],
            )
        )
        ind_x[i_exc, :] = -1
        ind_y[i_exc, :] = -1
        prf["ind_x_min"] = limits[2]  # reference back to original grid
        prf["ind_y_min"] = limits[0]

        ind_x_min = limits[2]
        ind_y_min = limits[0]

        # Sort 4 NN by distance on grid

        ip = np.where(np.logical_or(ind_x[:, 0] >= 0, ind_y[:, 0] >= 0))[0]

        lon_prf4 = np.repeat(lon_prf.values[ip, np.newaxis], 4, axis=1).ravel()
        lat_prf4 = np.repeat(lat_prf.values[ip, np.newaxis], 4, axis=1).ravel()
        r = np.ones(ind_x.shape) * np.nan
        # distance between nearest neighbors and grid
        rr = general_utils.calculate_haversine_distance(
            lon_prf4,
            lat_prf4,
            lon_grd.values[ind_y[ip, :].ravel(), ind_x[ip, :].ravel()],
            lat_grd.values[ind_y[ip, :].ravel(), ind_x[ip, :].ravel()],
        )

        r[ip, :] = np.reshape(rr, (ip.size, 4))
        # sort by distance and re-order the indices with closest first
        ii = np.argsort(r, axis=1)
        rmin_prf = np.take_along_axis(r, ii, axis=1)
        ind_x = np.take_along_axis(ind_x, ii, axis=1)
        ind_y = np.take_along_axis(ind_y, ii, axis=1)

        ii = np.nonzero(np.min(r, axis=1) > rmax)
        # Reference to original grid
        ind_x = ind_x + ind_x_min
        ind_y = ind_y + ind_y_min
        # mask bad values with -1
        ind_x[ii, :] = -1
        ind_y[ii, :] = -1
        ind_x[i_exc, :] = -1
        ind_y[i_exc, :] = -1
        # Add to profile object
        self.dataset["ind_x"] = xr.DataArray(ind_x, dims=["id_dim", "NNs"])
        self.dataset["ind_y"] = xr.DataArray(ind_y, dims=["id_dim", "NNs"])
        self.dataset["rmin_prf"] = xr.DataArray(rmin_prf, dims=["id_dim", "4"])

    def calculate_en4_qc_flags_levels(self):
        """
        Brute force method for identifying all rejected points according to
        EN4 binary integers. It can be slow to convert large numbers of integers
        to a sequence of bits and is actually quicker to just generate every
        combination of possible QC integers. That's what this routine does.
        Used in Profile.process_en4().

        INPUTS
         NO INPUTS

        OUTPUTS
         qc_integers_tem  : Array of integers signifying the rejection of ONLY
                            temperature datapoints
         qc_integers_sal  : Array of integers signifying the rejection of ONLY
                            salinity datapoints
         qc_integers_both : Array of integers signifying the rejection of BOTH
                            temperature and salinity datapoints.
        """

        reject_tem_ind = 0
        reject_sal_ind = 1
        if "4.2.0" in self.dataset.history:
            # from https://www.metoffice.gov.uk/hadobs/en4/en4-0-2-profile-file-format.html
            reject_tem_reasons = [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            reject_sal_reasons = [2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        elif "4.2.2" in self.dataset.history:
            # from https://www.metoffice.gov.uk/hadobs/en4/en4-2-2-profile-file-format.html
            reject_tem_reasons = [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            reject_sal_reasons = [2, 3, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        else:
            reject_tem_reasons = [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            reject_sal_reasons = [2, 3, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            debug(
                f"Assume QC flags following 4.2.2: https://www.metoffice.gov.uk/hadobs/en4/en4-2-2-profile-file-format.html"
            )

        qc_integers_tem = []
        qc_integers_sal = []
        qc_integers_both = []
        n_tem_reasons = len(reject_tem_reasons)
        n_sal_reasons = len(reject_sal_reasons)
        bin_len = 32

        # IF reject_tem = 1, reject_sal = 0
        for ii in range(n_tem_reasons):
            bin_tmp = np.zeros(bin_len, dtype=int)
            bin_tmp[reject_tem_ind] = 1
            bin_tmp[reject_tem_reasons[ii]] = 1
            qc_integers_tem.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))

        # IF reject_tem = 0, reject_sal = 1
        for ii in range(n_sal_reasons):
            bin_tmp = np.zeros(bin_len, dtype=int)
            bin_tmp[reject_sal_ind] = 1
            bin_tmp[reject_sal_reasons[ii]] = 1
            qc_integers_sal.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))

        # IF reject_tem = 1, reject_sal = 1
        for tt in range(n_tem_reasons):
            for ss in range(n_sal_reasons):
                bin_tmp = np.zeros(bin_len, dtype=int)
                bin_tmp[reject_tem_ind] = 1
                bin_tmp[reject_sal_ind] = 1
                bin_tmp[reject_tem_reasons[tt]] = 1
                bin_tmp[reject_sal_reasons[ss]] = 1
                qc_integers_both.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))

        qc_integers_tem = list(set(qc_integers_tem))
        qc_integers_sal = list(set(qc_integers_sal))
        qc_integers_both = list(set(qc_integers_both))

        return qc_integers_tem, qc_integers_sal, qc_integers_both

    """================Reshape to 2D================"""

    def reshape_2d(self, var_user_want):
        """
        OBSERVATION type class for reshaping World Ocean Data (WOD) or similar that
        contains 1D profiles (profile * depth levels)  into a 2D array.
        Note that its variable has its own dimention and in some profiles
        only some variables are present. WOD can be observed depth or a
        standard depth as regrided by NOAA.
           Args:
            > X     --      The variable (e.g,Temperatute, Salinity, Oxygen, DIC ..)
            > X_N    --     Dimensions of observed variable as 1D
                            (essentially number of obs variable = casts * osberved depths)
            > casts --      Dimension for locations of observations (ie. profiles)
            > z_N   --      Dimension for depth levels of all observations as 1D
                            (essentially number of depths = casts * osberved depths)
            > X_row_size -- Gives the vertical index (number of depths)
                            for each variable
        """

        """reshape the 1D variable into 2D variable (id_dimß, z_dim)    
        Args:
            profile       ::   The profile dimension. Called cast in WOD, 
                               common in all variables, but nans if 
                                a variable is not observed at a location
            z_dim         ::   The dimension for depth levels.
            var_user_want ::   List of observations the user wants to reshape       
        """

        # find maximum z levels in any of the profiles
        d_max = int(np.max(self.dataset.z_row_size.values))
        # number of profiles
        prof_size = self.dataset.z_row_size.shape[0]

        # set a 2D array (relevant to maximum depth)
        depth_2d = np.empty(
            (
                prof_size,
                d_max,
            )
        )
        depth_2d[:] = np.nan
        # reshape depth information from 1D to 2D
        if np.isnan(self.dataset.z_row_size.values[0]) == False:
            I_SIZE = int(self.dataset.z_row_size.values[0])
            depth_2d[0, :I_SIZE] = self.dataset.depth[0:I_SIZE].values
        for iJ in range(1, prof_size):
            if np.isnan(self.dataset.z_row_size.values[iJ]) == False:
                I_START = int(np.nansum(self.dataset.z_row_size.values[:iJ]))
                I_END = int(np.nansum(self.dataset.z_row_size.values[: iJ + 1]))
                I_SIZE = int(self.dataset.z_row_size.values[iJ])
                depth_2d[iJ, 0:I_SIZE] = self.dataset.depth[I_START:I_END].values

        # check reshape
        T_OBS1 = np.delete(np.ravel(depth_2d), np.isnan(np.ravel(depth_2d)))
        T_OBS2 = np.delete(self.dataset.depth.values, np.isnan(self.dataset.depth.values))
        if T_OBS1.size == T_OBS2.size and (int(np.min(T_OBS1 - T_OBS2)) == 0 or int(np.max(T_OBS1 - T_OBS2)) == 0):
            print("Depth OK reshape successful")
        else:
            print("Depth WRONG!! reshape")

        # reshape obs for each variable from 1D to 2D
        var_all = np.empty(
            (
                len(var_user_want),
                prof_size,
                d_max,
            )
        )
        var_list = var_user_want[:]
        counter_i = 0
        for iN in range(0, len(var_user_want)):
            print(var_user_want[iN])
            # check that variable exist in the WOD observations file
            if var_user_want[iN] in self.dataset:
                print("observed variable exist")
                # reshape it into 2D
                var_2d = np.empty(
                    (
                        prof_size,
                        d_max,
                    )
                )
                var_2d[:] = np.nan
                # populate array but make sure that the indexing for number of levels
                # is not nan, as in the data there are nan indexings for number of levels
                # indicating no observations there
                if np.isnan(self.dataset[var_user_want[iN] + "_row_size"][0].values) == False:
                    I_SIZE = int(self.dataset[var_user_want[iN] + "_row_size"][0].values)
                    var_2d[0, :I_SIZE] = self.dataset[var_user_want[iN]][0:I_SIZE].values
                for iJ in range(1, prof_size):
                    if np.isnan(self.dataset[var_user_want[iN] + "_row_size"].values[iJ]) == False:
                        I_START = int(np.nansum(self.dataset[var_user_want[iN] + "_row_size"].values[:iJ]))
                        I_END = int(np.nansum(self.dataset[var_user_want[iN] + "_row_size"].values[: iJ + 1]))
                        I_SIZE = int(self.dataset[var_user_want[iN] + "_row_size"].values[iJ])
                        var_2d[iJ, 0:I_SIZE] = self.dataset[var_user_want[iN]].values[I_START:I_END]

                # all variables in one array
                var_all[counter_i, :, :] = var_2d
                counter_i = counter_i + 1
                # check that you did everything correctly and the obs in yoru reshaped
                # array match the observations in original array
                T_OBS1 = np.delete(np.ravel(var_2d), np.isnan(np.ravel(var_2d)))
                del I_START, I_END, I_SIZE, var_2d
                T_OBS2 = np.delete(
                    self.dataset[var_user_want[iN]].values, np.isnan(self.dataset[var_user_want[iN]].values)
                )

                if T_OBS1.size == T_OBS2.size and (
                    int(np.min(T_OBS1 - T_OBS2)) == 0 or int(np.max(T_OBS1 - T_OBS2)) == 0
                ):
                    print("OK reshape successful")
                else:
                    print("WRONG!! reshape")

            else:
                print("variable not in observations")
                var_list[iN] = "NO"

        # REMOVE DUBLICATES
        var_list = list(dict.fromkeys(var_list))
        # REMOVE the non-observed variables from the list of variables
        var_list.remove("NO")

        # create the new 2D dataset array
        wod_profiles_2d = xr.Dataset(
            {
                "depth": (["id_dim", "z_dim"], depth_2d),
            },
            coords={
                "time": (["id_dim"], self.dataset.time.values),
                "latitude": (["id_dim"], self.dataset.latitude.values),
                "longitude": (["id_dim"], self.dataset.longitude.values),
            },
        )
        for iN in range(0, len(var_list)):
            wod_profiles_2d[var_list[iN]] = (["id_dim", "z_dim"], var_all[iN, :, :])

        return_prof = Profile()
        return_prof.dataset = wod_profiles_2d
        return return_prof

    def time_slice(self, date0, date1):
        """Return new Gridded object, indexed between dates date0 and date1"""
        dataset = self.dataset
        t_ind = pd.to_datetime(dataset.time.values) >= date0
        dataset = dataset.isel(id_dim=t_ind)
        t_ind = pd.to_datetime(dataset.time.values) < date1
        dataset = dataset.isel(id_dim=t_ind)
        return Profile(dataset=dataset)

    def calculate_vertical_spacing(self):
        """
        Profile data is given at depths, z, however for some calculations a thickness measure, dz, is required
        Define the upper thickness: dz[0] = 0.5*(z[0] + z[1]) and thereafter the centred difference:
        dz[k] = 0.5*(z[k-1] - z[k+1])

        Notionally, dz is the separation between w-points, when w-points are estimated from depths
        at t-points.
        """

        if hasattr(self.dataset, "dz"):  # Requires spacing variable. Test to see if variable exists
            pass
        else:
            # Compute dz on w-pts
            depth_t = self.dataset.depth
            self.dataset["dz"] = xr.where(
                depth_t == depth_t.min(dim="z_dim"),
                0.5 * (depth_t + depth_t.shift(z_dim=-1)),
                0.5 * (depth_t.shift(z_dim=-1) - depth_t.shift(z_dim=+1)),  # .fillna(0.)
            )
        attributes = {"units": "m", "standard name": "centre difference thickness"}
        self.dataset.dz.attrs = attributes

    def construct_density(
        self, eos="EOS10", rhobar=False, Zd_mask: xr.DataArray = None, CT_AS=False, pot_dens=False, Tbar=True, Sbar=True
    ):

        """
            Constructs the in-situ density using the salinity, temperature and
            depth fields. Adds a density attribute to the profile dataset

            Requirements: The supplied Profile dataset must contain the
            Practical Salinity and the Potential Temperature variables. The depth
            field must also be supplied. The GSW package is used to calculate
            The Absolute Pressure, Absolute Salinity and Conservative Temperature.

            Note that currently density can only be constructed using the EOS10
            equation of state.

        Parameters
        ----------
        eos : equation of state, optional
            DESCRIPTION. The default is 'EOS10'.

        rhobar : Calculate density with depth mean T and S
            DESCRIPTION. The default is 'False'.
        Zd_mask : (xr.DataArray) Provide a (id_dim, z_dim) mask for rhobar calculation
            Calculate using calculate_vertical_mask
            DESCRIPTION. The default is empty.

        CT_AS  : Conservative Temperature and Absolute Salinity already provided
            DESCRIPTION. The default is 'False'.
        pot_dens :Calculation at zero pressure
            DESCRIPTION. The default is 'False'.
        Tbar and Sbar : If rhobar is True then these can be switch to False to allow one component to
                        remain depth varying. So Tbar=Flase gives temperature component, Sbar=False gives Salinity component
            DESCRIPTION. The default is 'True'.

        Returns
        -------
        None.
        adds attribute profile.dataset.density

        """
        debug(f'Constructing in-situ density for {get_slug(self)} with EOS "{eos}"')

        try:

            if eos != "EOS10":
                raise ValueError(str(self) + ": Density calculation for " + eos + " not implemented.")

            try:
                shape_ds = (
                    self.dataset.id_dim.size,
                    self.dataset.z_dim.size,
                    # jth                    self.dataset.z_dim.size,
                    #                    self.dataset.id_dim.size,
                )
                sal = self.dataset.practical_salinity.to_masked_array()
                temp = self.dataset.potential_temperature.to_masked_array()

                if np.shape(sal) != shape_ds:
                    sal = sal.T
                    temp = temp.T
            except AttributeError:
                error(f"We have a problem with {self.dataset.dims}")

            density = np.ma.zeros(shape_ds)

            # print(f"shape sal:{np.shape(sal)}")
            # print(f"shape rho:{np.shape(density)}")

            s_levels = self.dataset.depth.to_masked_array()
            if np.shape(s_levels) != shape_ds:
                s_levels = s_levels.T

            lat = self.dataset.latitude.values
            lon = self.dataset.longitude.values
            # Absolute Pressure
            if pot_dens:
                pressure_absolute = 0.0  # calculate potential density
            else:
                pressure_absolute = np.ma.masked_invalid(gsw.p_from_z(-s_levels, lat))  # depth must be negative
            if not rhobar:  # calculate full depth
                # Absolute Salinity
                if not CT_AS:  # abs salinity not provided
                    sal_absolute = np.ma.masked_invalid(gsw.SA_from_SP(sal, pressure_absolute, lon, lat))
                else:  # abs salinity provided
                    sal_absolute = np.ma.masked_invalid(sal)
                sal_absolute = np.ma.masked_less(sal_absolute, 0)
                # Conservative Temperature
                if not CT_AS:  # conservative temp not provided
                    temp_conservative = np.ma.masked_invalid(gsw.CT_from_pt(sal_absolute, temp))
                else:  # conservative temp provided
                    temp_conservative = np.ma.masked_invalid(temp)
                # In-situ density
                density = np.ma.masked_invalid(gsw.rho(sal_absolute, temp_conservative, pressure_absolute))
                new_var_name = "density"
            else:  # calculate density with depth integrated T S

                if hasattr(self.dataset, "dz"):  # Requires spacing variable. Test to see if variable exists
                    pass
                else:  # Create it
                    self.calculate_vertical_spacing()

                # prepare coordinate variables
                if Zd_mask is None:
                    DZ = self.dataset.dz
                else:
                    DZ = self.dataset.dz * Zd_mask
                DP = DZ.sum(dim="z_dim").to_masked_array()
                DZ = DZ.to_masked_array()
                if np.shape(DZ) != shape_ds:
                    DZ = DZ.T
                # DP=np.repeat(DP[np.newaxis,:,:],shape_ds[1],axis=0)

                # DZ = np.repeat(DZ[np.newaxis, :, :, :], shape_ds[0], axis=0)
                # DP = np.repeat(DP[np.newaxis, :, :], shape_ds[0], axis=0)

                # Absolute Salinity
                if not CT_AS:  # abs salinity not provided
                    sal_absolute = np.ma.masked_invalid(gsw.SA_from_SP(sal, pressure_absolute, lon, lat))
                else:  # abs salinity provided
                    sal_absolute = np.ma.masked_invalid(sal)

                # Conservative Temperature
                if not CT_AS:  # Conservative temperature not provided
                    temp_conservative = np.ma.masked_invalid(gsw.CT_from_pt(sal_absolute, temp))
                else:  # conservative temp provided
                    temp_conservative = np.ma.masked_invalid(temp)

                if pot_dens and (Sbar and Tbar):  # usual case pot_dens and depth averaged everything
                    sal_absolute = np.sum(np.ma.masked_less(sal_absolute, 0) * DZ, axis=1) / DP
                    temp_conservative = np.sum(np.ma.masked_less(temp_conservative, 0) * DZ, axis=1) / DP
                    density = np.ma.masked_invalid(gsw.rho(sal_absolute, temp_conservative, pressure_absolute))
                    density = np.repeat(density[:, np.newaxis], shape_ds[1], axis=1)

                else:  # Either insitu density or one of Tbar or Sbar False
                    if Sbar:
                        sal_absolute = np.repeat(
                            (np.sum(np.ma.masked_less(sal_absolute, 0) * DZ, axis=1) / DP)[:, np.newaxis],
                            shape_ds[1],
                            axis=1,
                        )
                    if Tbar:
                        temp_conservative = np.repeat(
                            (np.sum(np.ma.masked_less(temp_conservative, 0) * DZ, axis=1) / DP)[:, np.newaxis],
                            shape_ds[1],
                            axis=1,
                        )
                    density = np.ma.masked_invalid(gsw.rho(sal_absolute, temp_conservative, pressure_absolute))

                if Tbar and Sbar:
                    new_var_name = "density_bar"

                else:
                    if not Tbar:
                        new_var_name = "density_T"
                    else:
                        new_var_name = "density_S"

            # rho and rhobar
            coords = {
                "time": (("id_dim"), self.dataset.time.values),
                "latitude": (("id_dim"), self.dataset.latitude.values),
                "longitude": (("id_dim"), self.dataset.longitude.values),
            }
            #            dims = ["z_dim", "id_dim"]
            dims = ["id_dim", "z_dim"]

            if pot_dens:
                attributes = {"units": "kg / m^3", "standard name": "Potential density "}
            else:
                attributes = {"units": "kg / m^3", "standard name": "In-situ density "}

            density = np.squeeze(density)
            self.dataset[new_var_name] = xr.DataArray(density, coords=coords, dims=dims, attrs=attributes)

        except AttributeError as err:
            error(err)

    def calculate_vertical_mask(self, Zmax=200):

        """
        Calculates a mask to a specified level Zmax. 1 for sea; 0 for below sea bed
        and linearly ramped for last level

        Inputs:
            Zmax float - max depth (m)
        Returns
        Zd_mask (id_dim, z_dim)  xr.DataArray, float mask.
        kmax (id_dim) deepest index above Zmax
        """

        depth_t = self.dataset.depth
        ##construct a W array, zero at surface 1/2 way between T-points

        depth_w = xr.zeros_like(depth_t)
        I = np.arange(depth_w.shape[1] - 1)
        depth_w[:, 0] = 0.0
        depth_w[:, I + 1] = 0.5 * (depth_t[:, I] + depth_t[:, I + 1])

        ## Contruct a mask array that is:
        # zeros below Zmax
        # ones above Zmax, except the closest shallower depth which has a value [0,1] that is the weighted distance to Zmax

        ## prepare depth profiles

        # remove deep nans
        # depth_t = depth_t.fillna(1E6)
        # depth_t = depth_t.interpolate_na(dim="z_dim", method="nearest", fill_value="extrapolate")
        # print(depth_t)

        ## construct a mask to identify location of and separation from Zmax

        # mask_arr = np.zeros((depth_t.shape))*np.nan
        # print(np.shape(mask_arr))
        # mask_arr[depth_t <= Zmax] = 1
        # mask_arr[depth_t > Zmax] = 0
        # mask = xr.DataArray( mask_arr, dims=["id_dim", "z_dim"])
        mask = depth_w * np.nan

        mask = xr.where(depth_w <= Zmax, 1, mask)
        mask = xr.where(depth_w > Zmax, 0, mask)

        # print(mask)
        # print('\n')

        max_shallower_depth = (depth_w * mask).max(dim="z_dim")
        min_deeper_depth = (depth_w.roll(z_dim=-1) * mask).max(dim="z_dim")
        # NB if max_shallower_depth was already deepest value in profile, then this produces the same value
        # I.e.
        # max_shallower_depth <= Zmax
        # min_deeper_depth > Zmax or min_deeper_depth = max_shallower_depth

        # print(f"max_shallower_depth:{max_shallower_depth}")
        # print(f"min_deeper_depth:{min_deeper_depth}")
        # print('\n')

        # Compute fraction, the relative closeness of Zmax to max_shallower_depth from 1 to 0 (as Zmax -> min_deeper_depth)
        fraction = xr.where(
            min_deeper_depth != max_shallower_depth,
            (Zmax - max_shallower_depth) / (min_deeper_depth - max_shallower_depth),
            1,
        )

        max_shallower_depth_2d = max_shallower_depth.expand_dims(dim={"z_dim": depth_w.sizes["z_dim"]})
        fraction_2d = fraction.expand_dims(dim={"z_dim": depth_t.sizes["z_dim"]})

        # locate the depth index for the deepest level above Zmax
        kmax = xr.where(depth_w == max_shallower_depth, 1, 0).argmax(dim="z_dim")
        # print(kmax)

        # replace mask values with fraction_2d at depth above Zmax)
        mask = xr.where(depth_w == max_shallower_depth_2d, fraction_2d, mask)

        return mask, kmax
