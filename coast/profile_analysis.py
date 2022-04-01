"""Profile Class"""
from .index import Indexed
import numpy as np
import xarray as xr
from . import general_utils, plot_util, Gridded, Profile
import matplotlib.pyplot as plt
import glob
import datetime
from .logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
import xarray.ufuncs as uf
import pandas as pd
from scipy import interpolate


class ProfileAnalysis(Indexed):
    """
    OBSERVATION type class for storing data ßfrom a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimension:

        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a
                     individual location.
        > z_dim   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files.
    """

    def __init__():
        """
        """

    """======================= Model Comparison ======================="""

    @classmethod
    def depth_means(cls, dataset, depth_bounds):
        """
        Calculates a mean of all variable data that lie between two depths.
        Returns a new Profile() object containing the meaned data

        INPUTS:
         dataset (Dataset)    : A dataset from a Profile object.
         depth_bounds (Tuple) : A tuple of length 2 describing depth bounds
                                Should be of form: (lower, upper) and in metres
        """

        debug(f"Averaging all variables between {0}m <= x < {1}m".format(depth_bounds[0], 
                                                                         depth_bounds[1]))
        # We need to remove any time variables or the later .where() won't work
        var_list = list(dataset.keys())
        time_var_list = ["time"]
        for vv in var_list:
            if dataset[vv].dtype in ["M8[ns]", "timedelta64[ns]"]:
                time_var_list.append(vv)

        # Extract/remove time vars from input dataset (but save them)
        time_vars = dataset[time_var_list]
        dataset = dataset.drop(time_var_list)

        # Get 2D indices of depth bounds, then indices of all points in the layer
        layer_ind0 = dataset.depth >= depth_bounds[0]
        layer_ind1 = dataset.depth < depth_bounds[1]
        layer_ind = layer_ind0 * layer_ind1
        
        # Mask out all other points
        masked = dataset.where(layer_ind, np.nan)
        
        # Calculate the average. By skipping nans, we only mean the points
        # within the desired layer
        meaned = masked.mean(dim="z_dim", skipna=True)

        # Remerge time variables and return new Profile object.
        ds = xr.merge((meaned, time_vars))
        return Profile(dataset = ds)

    @classmethod
    def bottom_means(cls, dataset, layer_thickness, depth_thresholds=[np.inf]):
        """
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

        # Extract bathymetry points and load them
        bathy_pts = dataset.bathymetry.values

        # We need to remove any time variables or the later .where() won't work
        var_list = list(dataset.keys())
        time_var_list = ["time"]
        for vv in var_list:
            if dataset[vv].dtype in ["M8[ns]", "timedelta64[ns]"]:
                time_var_list.append(vv)

        # Remove the time variables and save them for merge back later
        time_vars = dataset[time_var_list]
        dataset = dataset.drop(time_var_list)

        # A quick check to make sure the last threshold is infinite
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
            bottom_ind = dataset.depth >= (bathy_pts - prof_thickness)
        except:
            bottom_ind = dataset.depth.transpose() >= (bathy_pts - prof_thickness)
        dataset = dataset.where(bottom_ind, np.nan)

        # Average the remaining data
        dataset = dataset.mean(dim="z_dim", skipna=True)

        # Remerge time variables
        dataset = xr.merge((dataset, time_vars))
        return Profile(dataset=dataset)
    
    
    @classmethod
    def determine_mask_indices(cls, dataset, mask_dataset):
        """
        Determines whether each profile is within a mask (region) or not. 
        These masks should be in Dataset form, as returned by 
        Mask_maker().make_mask_dataset(). I.E, each mask
        should be a 2D array with corresponding 2D longitude and latitude 
        arrays. Multiple masks should be stored along a dim_mask dimension.

        Parameters
        ----------
        dataset : xarray.Dataset
            A dataset from a profile object
        mask_dataset : xarray.Dataset
            Dataset with dimensions (dim_mask, x_dim, y_dim). 
            Should contain longitude, latitude and mask. Mask has dimensions
            (dim_mask, y_dim, x_dim). Spatial dimensions should align with
            longitude and latitude

        Returns
        -------
        Dataset describing which profiles are in which mask/region.
        Ready for input to Profile.mask_means()
        """

        # If landmask not present, set to None for nearest_indices (no masking)
        if "landmask" not in list(mask_dataset.keys()):
            landmask = None
        else:
            landmask = mask_dataset.landmask
            
        # SPATIAL indices - nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2d(
            mask_dataset["longitude"], mask_dataset["latitude"], 
            dataset["longitude"], dataset["latitude"], mask=landmask
        )

        # Figure out which points lie in which region
        debug(f"Figuring out which regions each profile is in..")
        region_indices = mask_dataset.isel(x_dim=ind_x, y_dim=ind_y)

        return region_indices.rename({"dim_0": "id"})

    @classmethod
    def mask_means(cls, dataset, mask_indices):
        '''
        Averages all data inside a given profile dataset across a regional mask 
        or for multiples regional masks.

        Parameters
        ----------
        dataset : xarray.Dataset
            The profile dataset to average.
        mask_indices : xarray.Dataset
            Describes which profiles are in which region. Returned from 
            profile_analysis.determine_mask_indices().

        Returns
        -------
        xarray.Dataset containing meaned data. 

        '''

        # Get the number of masks provided
        n_masks = mask_indices.dims["dim_mask"]

        # Loop over maskal arrays. Assign mean to mask and seasonal means
        debug(f"Calculating maskal averages..")
        
        # Loop over masks and index the data
        for mm in range(0, n_masks):
            mask = mask_indices.isel(dim_mask=mm).mask.values
            mask_ind = np.where(mask.astype(bool))[0]
            if len(mask_ind) < 1:
                if mm == 0:
                    ds_average = xr.Dataset()
                continue
            
            # Get actual profile data for this mask
            mask_data = dataset.isel(profile=mask_ind)
            
            # Get two averages. One preserving depths and the other averaging
            # across all data in a region
            ds_average_prof = mask_data.mean(dim="profile", skipna=True).compute()
            ds_average_all = mask_data.mean(skipna=True).compute()

            # Loop over variables and save to output dataset.
            var_list = list(ds_average_prof.keys())
            for vv in var_list:
                ds_average_prof = ds_average_prof.rename({vv: "profile_mean_" + vv})
                ds_average_all = ds_average_all.rename({vv: "all_mean_" + vv})

            # Merge profile means and all means into one dataset for output
            ds_average_tmp = xr.merge((ds_average_prof, ds_average_all))

            # If only one mask then just return the merged dataset
            # else concatenate onto existing dataset.
            if mm == 0:
                ds_average = ds_average_tmp
            else:
                ds_average = xr.concat((ds_average, ds_average_tmp), dim="dim_mask")

        return Profile(dataset = ds_average)

    @classmethod
    def difference(cls, dataset1, dataset2, 
                   absolute_diff=True, square_diff=True):
        '''
        Calculates differences between all matched variables in two Profile 
        datasets. Difference direction is dataset1 - dataset2.

        Parameters
        ----------
        dataset1 : xarray.Dataset
            First profile dataset
        dataset2 : xarray.Dataset
            Second profile dataset
        absolute_diff : bool, optional
            Whether to calculate absolute differences. The default is True.
        square_diff : bool, optional
            Whether to calculate square differences. The default is True.

        Returns
        -------
        New Profile object containing differenced dataset.
        Differences have suffix diff_
        Absolute differences have suffix abs_
        Square differences have suffic square_

        '''
        # Difference the two input dataset
        differenced = dataset1 - dataset2
        
        # Get list of vars and coord names for later
        diff_vars = list(differenced.keys())
        save_coords = list(dataset1.coords.keys())

        # Loop over variables and rename differenced vars
        for vv in diff_vars:
            differenced = differenced.rename({vv: "diff_" + vv})

        # Loop over and rename absolutely differenced vars (if wanted)
        if absolute_diff:
            abs_tmp = np.fabs(differenced)
            diff_vars = list(abs_tmp.keys())
            for vv in diff_vars:
                abs_tmp = abs_tmp.rename({vv: "abs_" + vv})
        else:
            abs_tmp = xr.Dataset()

        # Loop over and rename square differenced vars (if wanted)
        if square_diff:
            sq_tmp = np.square(differenced)
            diff_vars = list(sq_tmp.keys())
            for vv in diff_vars:
                sq_tmp = sq_tmp.rename({vv: "square_" + vv})
        else:
            sq_tmp = xr.Dataset()

        # Merge the differences, absolute differences and square differences
        differenced = xr.merge((differenced, abs_tmp, sq_tmp, dataset1[save_coords]))
        return Profile(dataset=differenced)

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

            # interpolated_tmp = profile.interp(z_dim=new_depth_prof, method=interp_method)

            # interpolated_tmp = interpolated_tmp.rename_vars({"z_dim": "depth"})
            # interpolated_tmp = interpolated_tmp.reset_coords(["depth"])

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

    def average_into_grid_boxes(self, grid_lon, grid_lat, min_datapoints=1, season=None, var_modifier=""):
        """
        Takes the contents of this Profile() object and averages each variables
        into geographical grid boxes. At the moment, this expects there to be
        no vertical dimension (z_dim), so make sure to slice the data out you
        want first using isel, Profile.depth_means() or Profile.bottom_means().

        INPUTS
         grid_lon (array)     : 1d array of longitudes
         grid_lat (array)     : 1d array of latitude
         min_datapoints (int) : Minimum N of datapoints at which to average
                                into box. Will return Nan in boxes with smaller N.
                                NOTE this routine will also return the variable
                                grid_N, which tells you how many points were
                                averaged into each box.
        season (str)          : 'DJF','MAM','JJA' or 'SON'. Will only average
                                data from specified season.
        var_modifier (str)    : Suffix to add to all averaged variables in the
                                output dataset. For example you may want to add
                                _DJF to all vars if restricting only to winter.

        OUTPUTS
         COAsT Gridded object containing averaged data.
        """

        # Get the dataset in this object
        ds = self.dataset

        # Get a list of variables in this dataset
        vars_in = [items for items in ds.keys()]
        vars_out = [vv + "{0}".format(var_modifier) for vv in vars_in]

        # Get output dimensions and create 2D longitude and latitude arrays
        n_r = len(grid_lat) - 1
        n_c = len(grid_lon) - 1
        lon_mids = (grid_lon[1:] + grid_lon[:-1]) / 2
        lat_mids = (grid_lat[1:] + grid_lat[:-1]) / 2
        lon2, lat2 = np.meshgrid(lon_mids, lat_mids)

        # Create empty output dataset
        ds_out = xr.Dataset(coords=dict(longitude=(["y_dim", "x_dim"], lon2), latitude=(["y_dim", "x_dim"], lat2)))

        # Loop over variables and create empty placeholders
        for vv in vars_out:
            ds_out[vv] = (['y_dim','x_dim'], np.zeros((n_r, n_c))*np.nan)
        # Grid_N is the count ineach box
        ds_out["grid_N{0}".format(var_modifier)] = (["y_dim", "x_dim"], np.zeros((n_r, n_c)) * np.nan)

        # Extract season if needed
        if season is not None:
            season_array = general_utils.determine_season(ds.time)
            s_ind = season_array == season
            ds = ds.isel(profile=s_ind)

        # Loop over every box (slow??)
        for rr in range(n_r - 1):
            for cc in range(n_c - 1):
                # Get box bounds for easier understanding
                lon_min = grid_lon[cc]
                lon_max = grid_lon[cc + 1]
                lat_min = grid_lat[rr]
                lat_max = grid_lat[rr + 1]

                # Get profiles inside this box
                condition1 = np.logical_and(ds.longitude >= lon_min, ds.longitude < lon_max)
                condition2 = np.logical_and(ds.latitude >= lat_min, ds.latitude < lat_max)
                prof_ind = np.logical_and(condition1, condition2)

                # Only average if N > min_datapoints
                if np.sum(prof_ind) > min_datapoints:
                    for vv in range(len(vars_in)):
                        vv_in = vars_in[vv]
                        vv_out = vars_out[vv]
                        ds_out[vv_out][rr, cc] = ds[vv_in].isel(profile=prof_ind).mean()
                        #ds_out[vv_out][rr, cc] = np.nanmean(ds[vv_in].isel(profile=prof_ind))

                # Store N in own variable
                ds_out["grid_N{0}".format(var_modifier)][rr, cc] = np.sum(prof_ind)

        # Create and populate output dataset
        gridded_out = Gridded()
        gridded_out.dataset = ds_out

        return gridded_out
