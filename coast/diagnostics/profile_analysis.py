"""Profile Class"""
from ..data.index import Indexed
import numpy as np
import xarray as xr
from .._utils import general_utils, plot_util
from ..data.gridded import Gridded
from ..data.profile import Profile
from .._utils.logging_util import get_slug, debug, info, warn, warning
from scipy import interpolate


class ProfileAnalysis(Indexed):
    """
    A set of analysis routines suitable for datasets in a Profile object.
    See individual docstrings in each method for more info.
    """

    def __init__(self):
        """ """

    """======================= Model Comparison ======================="""

    @classmethod
    def depth_means(cls, profile, depth_bounds):
        """
        Calculates a mean of all variable data that lie between two depths.
        Returns a new Profile() object containing the meaned data

        INPUTS:
         dataset (Dataset)    : A dataset from a Profile object.
         depth_bounds (Tuple) : A tuple of length 2 describing depth bounds
                                Should be of form: (lower, upper) and in metres
        """

        debug(f"Averaging all variables between {0}m <= x < {1}m".format(depth_bounds[0], depth_bounds[1]))
        dataset = profile.dataset

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
        return Profile(dataset=ds)

    @classmethod
    def bottom_means(cls, profile, layer_thickness, depth_thresholds=[np.inf]):
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
        dataset = profile.dataset

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
    def determine_mask_indices(cls, profile, mask_dataset):
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
        dataset = profile.dataset
        # If landmask not present, set to None for nearest_indices (no masking)
        if "landmask" not in list(mask_dataset.keys()):
            landmask = None
        else:
            landmask = mask_dataset.landmask

        # SPATIAL indices - nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2d(
            mask_dataset["longitude"],
            mask_dataset["latitude"],
            dataset["longitude"],
            dataset["latitude"],
            mask=landmask,
        )

        # Figure out which points lie in which region
        debug(f"Figuring out which regions each profile is in..")
        region_indices = mask_dataset.isel(x_dim=ind_x, y_dim=ind_y)

        return region_indices.rename({"dim_0": "id_dim"})

    @classmethod
    def mask_means(cls, profile, mask_indices):
        """
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

        """
        dataset = profile.dataset
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
            mask_data = dataset.isel(id_dim=mask_ind)
            # Get two averages. One preserving depths and the other averaging
            # across all data in a region
            ds_average_prof = mask_data.mean(dim="id_dim", skipna=True).compute()
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

        return ds_average

    @classmethod
    def difference(cls, profile1, profile2, absolute_diff=True, square_diff=True):
        """
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

        """
        dataset1 = profile1.dataset
        dataset2 = profile2.dataset

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

    @classmethod
    def interpolate_vertical(cls, profile, new_depth, interp_method="linear", print_progress=False):
        """
        (04/10/2021)
        Author: David Byrne

        For vertical interpolation of all profiles within this object. User
        should pass an array describing the new depths or another profile object
        containing the same number of profiles as this object.

        If a 1D numpy array is passed then all profiles will be interpolated
        onto this single set of depths. If a xarray.DataArray is passed, it
        should have dimensions (id_dim, z_dim) and contain a variable called
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

        ds = profile.dataset
        n_prof = ds.sizes["id_dim"]
        n_z = ds.sizes["z_dim"]

        # Get variable names on z_dim dimension
        zvars = []
        notzvars = []
        for items in ds.keys():
            if "z_dim" in ds[items].dims:
                zvars.append(items)
            else:
                notzvars.append(items)

        # Now loop over profiles and interpolate model onto obs.
        count_ii = 0
        for pp in range(0, n_prof):
            if print_progress:
                print("{0} / {1}".format(pp, n_prof - 1))

            # Select the current profile
            profile = ds.isel(id_dim=pp)  # .rename({"depth": "z_dim"})

            # Extract new depths for this profile
            if repeated_depth:
                new_depth_prof = new_depth
            else:
                new_depth_prof = new_depth.isel(id_dim=pp).values

            # Do the interpolation and rename dimensions/vars back to normal
            interpolated_tmp = profile[notzvars]

            for vv in zvars:
                if vv == "depth":
                    continue
                # Get arrays to interpolate
                interpx = profile.depth.values
                interpy = profile[vv].values

                # Remove NaNs
                xnans = np.isnan(interpx)
                ynans = np.isnan(interpy)
                xynans = np.logical_or(xnans, ynans)
                interpx = interpx[~xynans]
                interpy = interpy[~xynans]

                # If there are <2 datapoints, dont interpolate. Return a nan
                # array instead for this profile variable
                if len(interpx) >= 2:
                    # Use scipy to interpolate this profile
                    interp_func = interpolate.interp1d(
                        interpx, interpy, bounds_error=False, kind=interp_method, fill_value=np.nan
                    )
                    vv_interp = interp_func(new_depth_prof)
                    interpolated_tmp[vv] = ("z_dim", vv_interp)
                else:
                    interpolated_tmp[vv] = ("z_dim", np.zeros(len(new_depth_prof)) * np.nan)

            # Put the new depth into the interpolated profile
            interpolated_tmp["depth"] = ("z_dim", new_depth_prof)

            # If not first iteration, concat this interpolated profile
            if count_ii == 0:
                interpolated = interpolated_tmp
            else:
                interpolated = xr.concat((interpolated, interpolated_tmp), dim="id_dim", coords="all")
            count_ii = count_ii + 1

        # Set depth to be a coordinate and return a new Profile object.
        interpolated = interpolated.set_coords(["depth"])
        return Profile(dataset=interpolated)

    @classmethod
    def average_into_grid_boxes(cls, profile, grid_lon, grid_lat, min_datapoints=1, season=None, var_modifier=""):
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
        ds = profile.dataset

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
            ds_out[vv] = (["y_dim", "x_dim"], np.zeros((n_r, n_c)) * np.nan)
        # Grid_N is the count ineach box
        ds_out["grid_N{0}".format(var_modifier)] = (["y_dim", "x_dim"], np.zeros((n_r, n_c)) * np.nan)

        # Extract season if needed
        if season is not None:
            season_array = general_utils.determine_season(ds.time)
            s_ind = season_array == season
            ds = ds.isel(id_dim=s_ind)

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

                        # Check datatype isn't string
                        if ds[vv_in].dtype in ["S1", "S2"]:
                            continue
                        ds_out[vv_out][rr, cc] = ds[vv_in].isel(id_dim=prof_ind).mean()

                # Store N in own variable
                ds_out["grid_N{0}".format(var_modifier)][rr, cc] = np.sum(prof_ind)

        # Create and populate output dataset
        gridded_out = Gridded()
        gridded_out.dataset = ds_out

        return gridded_out
