"""Profile Class"""
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
import scipy.interpolate as interpolate


class Profile(Indexed):
    """
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimension:

        > id      :: The profiles dimension. Each element of this dimension 
                     contains data (e.g. cast) for an individual location.
        > z_dim   :: The dimension for depth levels. A profile object does not
                     need to have shared depths, so NaNs might be used to
                     pad any depth array.
    
    Alongside these dimensions, the following minimal coordinates should also
    be available:
        
        > longitude (id)   :: 1D array of longitudes, one for each id
        > latitude  (id)   :: 1D array of latitudes, one for each id
        > time      (id)   :: 1D array of times, one for each id
        > depth     (id, z_dim)  :: 2D array of depths, with different depth
                                    levels being provided for each profile.
                                    Note that these depth levels need to be
                                    stored in a 2D array, so NaNs can be used
                                    to pad out profiles with shallower depths.
        > id_name   (id)   :: [Optional] Name of id/case or id number.
                                    
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

    def __init__(self, dataset = None, config: Union[Path, str] = None):
        """Initialization and file reading. You may initialize 

        Args:
            config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(config)
        
        # If dataset is provided, put inside this object
        if dataset is not None:
            self.dataset = dataset
            self.apply_config_mappings()

        debug(f"{get_slug(self)} initialised")

    def read_en4(self, fn_en4, chunks: dict = {}, multiple=False) -> None:
        """Reads a single or multiple EN4 netCDF files into the COAsT profile 
        data structure.

        Args:
            fn_en4 (str): path to data file
            chunks (dict): chunks
            multiple (boolean): True if reading multiple files otherwise False
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
            dates = [ff[-9:-3] for ff in file_to_read]
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

    """======================= Manipulate ======================="""

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """ Get a subset of this Profile() object in a spatial box.

        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]

        return: A new profile object containing subsetted data
        """
        ind = general_utils.subset_indices_lonlat_box(
            self.dataset.longitude, self.dataset.latitude, lonbounds[0], 
            lonbounds[1], latbounds[0], latbounds[1]
        )
        return Profile(dataset = self.dataset.isel(id=ind))

    """======================= Plotting ======================="""

    def plot_profile(self, var: str, profile_indices=None):

        fig = plt.figure(figsize=(7, 10))

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims["profile"])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(profile=ii)
            prof_depth = self.dataset.depth.isel(profile=ii)
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

        profile = self.dataset.isel(profile=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)

        return fig, ax

    """======================= Model Comparison ======================="""

    def process_en4(self, sort_time=True):
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
        qc_str = [np.binary_repr(ds.qc_flags_profiles.values[pp]).zfill(30)[::-1] for pp in range(ds.dims["profile"])]

        # Determine indices of the profiles that we want to keep
        reject_tem_prof = np.array([int(qq[0]) for qq in qc_str], dtype=bool)
        reject_sal_prof = np.array([int(qq[1]) for qq in qc_str], dtype=bool)
        reject_both_prof = np.logical_and(reject_tem_prof, reject_sal_prof)
        ds["reject_tem_prof"] = (["profile"], reject_tem_prof)
        ds["reject_sal_prof"] = (["profile"], reject_sal_prof)
        debug(
            "     >>> QC: Completely rejecting {0} / {1} profiles".format(np.sum(reject_both_prof), ds.dims["profile"])
        )

        ds = ds.isel(profile=~reject_both_prof)
        reject_tem_prof = reject_tem_prof[~reject_both_prof]
        reject_sal_prof = reject_sal_prof[~reject_both_prof]
        qc_lev = ds.qc_flags_levels.values

        debug(f" QC: Additional profiles converted to NaNs: ")
        debug(f"     >>> {0} temperature profiles ".format(np.sum(reject_tem_prof)))
        debug(f"     >>> {0} salinity profiles ".format(np.sum(reject_sal_prof)))

        reject_tem_lev = np.zeros((ds.dims["profile"], ds.dims["z_dim"]), dtype=bool)
        reject_sal_lev = np.zeros((ds.dims["profile"], ds.dims["z_dim"]), dtype=bool)

        int_tem, int_sal, int_both = self.calculate_all_en4_qc_flags()
        for ii in range(len(int_tem)):
            reject_tem_lev[qc_lev == int_tem[ii]] = 1
        for ii in range(len(int_sal)):
            reject_sal_lev[qc_lev == int_sal[ii]] = 1
        for ii in range(len(int_both)):
            reject_tem_lev[qc_lev == int_both[ii]] = 1
            reject_sal_lev[qc_lev == int_both[ii]] = 1

        ds["reject_tem_datapoint"] = (["profile", "z_dim"], reject_tem_lev)
        ds["reject_sal_datapoint"] = (["profile", "z_dim"], reject_sal_lev)

        debug(f"MASKING rejected datapoints, replacing with NaNs...")
        ds["temperature"] = xr.where(~reject_tem_lev, ds["temperature"], np.nan)
        ds["potential_temperature"] = xr.where(~reject_tem_lev, ds["temperature"], np.nan)
        ds["practical_salinity"] = xr.where(~reject_tem_lev, ds["practical_salinity"], np.nan)

        if sort_time:
            debug(f"Sorting Time Dimension...")
            ds = ds.sortby("time")

        debug(f"Finished processing data. Returning new Profile object.")

        return_prof = Profile()
        return_prof.dataset = ds
        return return_prof

    @classmethod
    def calculate_all_en4_qc_flags(cls):
        """
        Brute force method for identifying all rejected points according to
        EN4 binary integers. It can be slow to convert large numbers of integers
        to a sequence of bits and is actually quicker to just generate every
        combination of possible QC integers. That's what this routine does.
        Used in PROFILE.preprocess_en4().

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
        reject_tem_reasons = [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        reject_sal_reasons = [2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        qc_integers_tem = []
        qc_integers_sal = []
        qc_integers_both = []
        n_tem_reasons = len(reject_tem_reasons)
        n_sal_reasons = len(reject_sal_reasons)
        bin_len = 30

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
    
    def interpolate_vertical(self, new_depth, interp_method="linear"):
        """
        (04/10/2021)
        Author: David Byrne

        For vertical interpolation of all profiles within this object. User
        should pass an array describing the new depths or another profile object
        containing the same number of profiles as this object.

        If a 1D numpy array is passed then all profiles will be interpolated
        onto this single set of depths. If a xarray.DataArray is passed, it
        should have dimensions (profile, z_dim) and contain a variable called
        depth. This DataArray should contain the same number of profiles as
        this object and will map profiles in order for interpolation. If
        another profile object is passed, profiles will be mapped and
        interpolated onto the other objects depth array.

        INPUTS:
         new_depth (array or dataArray) : new depths onto which to interpolate
                                          see description above for more info.
         interp_method (str)            : Any scipy interpolation string.

        OUTPUTS:
         Returns a new PROFILE object containing the interpolated dataset.
        """

        # If input is Profile, extract depth dataarray
        if type(new_depth) is Profile:
            new_depth = new_depth.dataset.depth

        # If input is 1D, then interpolation will be done onto this for all.
        if len(new_depth.shape) == 1:
            repeated_depth = True
            debug(f"Interpolating onto reference depths")
        else:
            debug(f"Interpolating onto depths of existing Profile object")
            repeated_depth = False

        ds = self.dataset
        n_prof = ds.dims["profile"]

        # Get variable names on z_dim dimension
        zvars = []
        notzvars = []
        for items in ds.keys():
            print(items)
            if "z_dim" in ds[items].dims:
                zvars.append(items)
            else:
                notzvars.append(items)

        # Now loop over profiles and interpolate model onto obs.
        count_ii = 0
        for pp in range(0, n_prof):
            print("{0} / {1}".format(pp, n_prof))

            # Select the current profile
            profile = ds.isel(profile=pp)  # .rename({"depth": "z_dim"})
            # profile = profile.dropna("z_dim")
            # profile = profile.set_coords("depth")

            # Extract new depths for this profile
            if repeated_depth:
                new_depth_prof = new_depth
            else:
                new_depth_prof = new_depth.isel(profile=pp).values

            # Do the interpolation and rename dimensions/vars back to normal
            interpolated_tmp = profile[notzvars]

            for vv in zvars:
                if vv == "depth":
                    continue
                print(vv)
                interpx = profile.depth.values
                interpy = profile[vv].values
                interp_func = interpolate.interp1d(
                    interpx, interpy, bounds_error=False, kind=interp_method, fill_value=np.nan
                )
                vv_interp = interp_func(new_depth_prof)
                interpolated_tmp[vv] = ("z_dim", vv_interp)

            interpolated_tmp["depth"] = ("z_dim", new_depth_prof)

            # If not first iteration, concat this interpolated profile
            if count_ii == 0:
                interpolated = interpolated_tmp
            else:
                interpolated = xr.concat((interpolated, interpolated_tmp), dim="profile", coords="all")
            count_ii = count_ii + 1

        # Create and format output dataset
        interpolated = interpolated.set_coords(["depth"])
        return Profile(dataset=interpolated)

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
        ind_t = [np.argmin(np.abs(mod_time - en4_time[tt])) for tt in range(en4.dims["profile"])]
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
        time_chunks = gridded.chunks["t_dim"]
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
                count_ii = count_ii + 1
                if count_ii == 0:
                    mod_profiles = xr.Dataset()
                continue

            # Pull out x,y and t indices
            ind_x_in_chunk = ind_x[ind_in_chunk]
            ind_y_in_chunk = ind_y[ind_in_chunk]
            ind_t_in_chunk = ind_t[ind_in_chunk] - start_ii

            # Index a temporary chunk and read it to memory
            ds_tmp = gridded.isel(t_dim=np.arange(start_ii, end_ii)).load()

            # Index loaded chunk and rename dim_0 to profile
            ds_tmp_indexed = ds_tmp.isel(x_dim=ind_x_in_chunk, y_dim=ind_y_in_chunk, t_dim=ind_t_in_chunk)
            ds_tmp_indexed = ds_tmp_indexed.rename({"dim_0": "profile"})

            # Mask out all levels deeper than bottom_level
            # Here I have used set_coords() and reset_coords() to omit variables
            # with no z_dim from the masking. Otherwise xr.where expands these
            # dimensions into full 2D arrays.
            if mask_bottom_level:
                n_z_tmp = ds_tmp_indexed.dims["z_dim"]
                bl_array = ds_tmp_indexed.bottom_level.values
                z_index, bl_index = np.meshgrid(np.arange(0, n_z_tmp), bl_array)
                mask2 = xr.DataArray(z_index < bl_index, dims=["profile", "z_dim"])
                ds_tmp_indexed = ds_tmp_indexed.set_coords(bl_var_list)
                ds_tmp_indexed = ds_tmp_indexed.where(mask2)
                ds_tmp_indexed = ds_tmp_indexed.reset_coords(bl_var_list)

            # If not first iteration, concatenate this indexed chunk onto
            # final output dataset
            if count_ii == 0:
                mod_profiles = ds_tmp_indexed
            else:
                mod_profiles = xr.concat((mod_profiles, ds_tmp_indexed), dim="profile")

            # Update counters
            start_ii = end_ii
            count_ii = count_ii + 1

        # Put obs time into the output array
        mod_profiles["obs_time"] = (["profile"], en4_time)

        # Calculate interpolation distances
        interp_dist = general_utils.calculate_haversine_distance(
            en4.longitude, en4.latitude, mod_profiles.longitude, mod_profiles.latitude
        )
        mod_profiles["interp_dist"] = (["profile"], interp_dist.values)

        # Calculate interpolation time lags
        interp_lag = (mod_profiles.time.values - en4_time).astype("timedelta64[h]")
        mod_profiles["interp_lag"] = (["profile"], interp_lag)

        # Put x and y indices into dataset
        mod_profiles["nearest_index_x"] = (["profile"], ind_x.values)
        mod_profiles["nearest_index_y"] = (["profile"], ind_y.values)
        mod_profiles["nearest_index_t"] = (["profile"], ind_t.values)

        # Create return object and put dataset into it.
        return_prof = Profile()
        return_prof.dataset = mod_profiles
        return return_prof
    

            
            
            
            
            
            
            
            
            
