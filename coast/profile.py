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
import xarray.ufuncs as uf
import pandas as pd


class Profile(Indexed):
    """
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimension:

        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a
                     individual location.
        > z_dim   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files.
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
        ind = general_utils.subset_indices_lonlat_box(
            self.dataset.longitude, self.dataset.latitude, lonbounds[0], lonbounds[1], latbounds[0], latbounds[1]
        )
        return ind

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

    def depth_means(self, depth_bounds):
        """
        (04/10/2021)
        Author: David Byrne

        INPUTS:
         masks (Dataset) :

        OUTPUTS:

        """

        debug(f"Averaging all variables between {0}m <= x < {1}m".format(depth_bounds[0], depth_bounds[1]))
        ds = self.dataset

        # We need to remove any time variables or the later .where() won't work
        var_list = list(ds.keys())
        time_var_list = ["time"]
        for vv in var_list:
            if ds[vv].dtype in ["M8[ns]", "timedelta64[ns]"]:
                time_var_list.append(vv)

        time_vars = ds[time_var_list]
        ds = ds.drop(time_var_list)

        layer_ind0 = ds.depth >= depth_bounds[0]
        layer_ind1 = ds.depth < depth_bounds[1]
        layer_ind = layer_ind0 * layer_ind1
        masked = ds.where(layer_ind, np.nan)
        meaned = masked.mean(dim="z_dim", skipna=True).load()

        # Remerge time variables
        ds = xr.merge((meaned, time_vars))

        return_prof = Profile()
        return_prof.dataset = ds

        return return_prof

    def bottom_means(self, layer_thickness, depth_thresholds=[np.inf]):
        """
        (04/10/2021)
        Author: David Byrne

        Averages profile data in some layer above the bathymetric depth. This
        routine requires there to be a 'bathymetry' variable in the Profile dataset.
        It can apply a constant averaging layer thickness across all profiles
        or a bespoke thickness dependent on the bathymetric depth. For example,
        you may want to define the 'bottom' as the average of 100m above the
        bathymetry in very deep ocean but only 10m in the shallower ocean.
        If there is no data available in the layer specified (e.g. CTD cast not
        deep enough or model bathymetry wrong) then it will be NaN

        To apply constant thickness, you only need to provide a value (in metre)
        for layer_thickness. For different thicknesses, you also need to give
        depth_thresholds. The last threshold must always be np.inf, i.e. all
        data below a specific bathymetry depth.

        For example, to apply 10m to everywhere <100m, 50m to 100m -> 500m and
        100m elsewhere, use:

            layer_thickness = [10, 50, 100]
            depth_thresholds = [100, 500, np.inf]

        The bottom bound is always assumed to be 0.

        *NOTE: If time related issues arise, then remove any time variables
        from the profile dataset before running this routine.

        INPUTS:
         layer_thickness (array) : A scalar layer thickness or list of values
         depth_thresholds (array) : Optional. List of bathymetry thresholds.

        OUTPUTS:
         New profile object containing bottom averaged data.

        """

        # Extract bathymetry points
        ds = self.dataset
        bathy_pts = ds.bathymetry.values

        # We need to remove any time variables or the later .where() won't work
        var_list = list(ds.keys())
        time_var_list = ["time"]
        for vv in var_list:
            if ds[vv].dtype in ["M8[ns]", "timedelta64[ns]"]:
                time_var_list.append(vv)

        time_vars = ds[time_var_list]
        ds = ds.drop(time_var_list)

        if depth_thresholds[-1] != np.inf:
            raise ValueError("Please ensure the final element of depth_thresholds is np.inf")

        # Convert to numpy arrays where necessary
        if np.isscalar(layer_thickness):
            layer_thickness = [layer_thickness]

        depth_thresholds = np.array(depth_thresholds)
        layer_thickness = np.array(layer_thickness)

        # Get depths, thresholds and thicknesses at each profile
        prof_threshold = np.array([np.sum(ii > depth_thresholds) for ii in bathy_pts])
        prof_thickness = layer_thickness[prof_threshold]

        # Insert NaNs where not in the bottom
        try:
            bottom_ind = ds.depth >= (bathy_pts - prof_thickness)
        except:
            bottom_ind = ds.depth.transpose() >= (bathy_pts - prof_thickness)
        ds = ds.where(bottom_ind, np.nan)

        # Average the remaining data
        ds = ds.mean(dim="z_dim", skipna=True).load()

        # Remerge time variables
        ds = xr.merge((ds, time_vars))

        return_prof = Profile()
        return_prof.dataset = ds

        return return_prof

    def determine_mask_indices(self, masks):
        """
        (04/10/2021)
        Author: David Byrne

        INPUTS:
         masks (Dataset) :

        OUTPUTS:

        """

        ds = self.dataset

        # If landmask not present, set to None for nearest_indices (no masking)
        if "landmask" not in list(masks.keys()):
            landmask = None
        else:
            landmask = masks.landmask
        # SPATIAL indices - nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2d(
            masks["longitude"], masks["latitude"], ds["longitude"], ds["latitude"], mask=landmask
        )

        # Figure out which points lie in which region
        debug(f"Figuring out which regions each profile is in..")
        region_indices = masks.isel(x_dim=ind_x, y_dim=ind_y)

        return region_indices.rename({"dim_0": "profile"})

    def mask_means(self, mask_indices):

        ds = self.dataset

        n_masks = mask_indices.dims["dim_mask"]

        # Loop over maskal arrays. Assign mean to mask and seasonal means
        debug(f"Calculating maskal averages..")
        for mm in range(0, n_masks):
            mask = mask_indices.isel(dim_mask=mm).mask.values
            mask_ind = np.where(mask.astype(bool))[0]
            if len(mask_ind) < 1:
                if mm == 0:
                    ds_average = xr.Dataset()
                continue
            mask_data = ds.isel(profile=mask_ind)
            ds_average_prof = mask_data.mean(dim="profile", skipna=True).compute()
            ds_average_all = mask_data.mean(skipna=True).compute()

            var_list = list(ds_average_prof.keys())
            for vv in var_list:
                ds_average_prof = ds_average_prof.rename({vv: "profile_average_" + vv})
                ds_average_all = ds_average_all.rename({vv: "average_" + vv})

            ds_average_tmp = xr.merge((ds_average_prof, ds_average_all))

            if mm == 0:
                ds_average = ds_average_tmp
            else:
                ds_average = xr.concat((ds_average, ds_average_tmp), dim="dim_mask")

        return ds_average

    def difference(self, other, absolute_diff=True, square_diff=True):

        differenced = self.dataset - other.dataset
        diff_vars = list(differenced.keys())
        save_coords = list(self.dataset.coords.keys())

        for vv in diff_vars:
            differenced = differenced.rename({vv: "diff_" + vv})

        if absolute_diff:
            abs_tmp = uf.fabs(differenced)
            diff_vars = list(abs_tmp.keys())
            for vv in diff_vars:
                abs_tmp = abs_tmp.rename({vv: "abs_" + vv})
        else:
            abs_tmp = xr.Dataset()

        if square_diff:
            sq_tmp = uf.square(differenced)
            diff_vars = list(sq_tmp.keys())
            for vv in diff_vars:
                sq_tmp = sq_tmp.rename({vv: "square_" + vv})
        else:
            sq_tmp = xr.Dataset()

        differenced = xr.merge((differenced, abs_tmp, sq_tmp, self.dataset[save_coords]))

        return_differenced = Profile()
        return_differenced.dataset = differenced

        return return_differenced

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

        # Now loop over profiles and interpolate model onto obs.
        count_ii = 0
        for pp in range(0, n_prof):

            # Select the current profile
            profile = ds.isel(profile=pp).rename({"depth": "z_dim"})
            profile = profile.dropna("z_dim")
            profile = profile.set_coords("z_dim")

            # Extract new depths for this profile
            if repeated_depth:
                new_depth_prof = new_depth
            else:
                new_depth_prof = new_depth.isel(profile=pp).values

            # Do the interpolation and rename dimensions/vars back to normal
            interpolated_tmp = profile.interp(z_dim=new_depth_prof, method=interp_method)

            interpolated_tmp = interpolated_tmp.rename_vars({"z_dim": "depth"})
            interpolated_tmp = interpolated_tmp.reset_coords(["depth"])

            # If not first iteration, concat this interpolated profile
            if count_ii == 0:
                interpolated = interpolated_tmp
            else:
                interpolated = xr.concat((interpolated, interpolated_tmp), dim="profile", coords="all")
            count_ii = count_ii + 1

        # Create and format output dataset
        interpolated = interpolated.set_coords(["depth"])
        return_interpolated = Profile()
        return_interpolated.dataset = interpolated

        return return_interpolated

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

        # REJECT profiles that are QC flagged.
        debug(f" Applying QUALITY CONTROL to EN4 data...")
        ds.qc_flags_profiles.load()

        # This line reads converts the QC integer to a binary string.
        # Each bit of this string is a different QC flag. Which flag is which can
        # be found on the EN4 website:
        # https://www.metoffice.gov.uk/hadobs/en4/en4-0-2-profile-file-format.html
        qc_str = [np.binary_repr(ds.qc_flags_profiles.values[pp]).zfill(30)[::-1] for pp in range(ds.dims["profile"])]

        # Determine indices of kept profiles
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

    def calculate_all_en4_qc_flags(self):
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
