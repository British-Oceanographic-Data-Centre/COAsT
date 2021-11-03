import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import pytz
import sklearn.metrics as metrics
from . import general_utils, plot_util, crps_util, stats_util
from .logging_util import get_slug, debug, error, info
import xarray.ufuncs as uf
import matplotlib.dates as mdates
import utide as ut
import scipy.signal as signal


class TidegaugeMultiple:
    """
    This is an object for storage and manipulation of multiple tide gauges
    in a single dataset. This may require some processing of the observations
    such as interpolation to a common time step.

    This object's dataset should take the form:

        Dimensions:
            port : The locations dimension. Each time series has an index
            time : The time dimension. Each datapoint at each port has an index

        Coordinates:
            longitude (port) : Longitude values for each port index
            latitude  (port) : Latitude values for each port index
            time      (port) : Time values for each time index (datetime)

    An example data variable could be ssh, or ntr (non-tidal residual). This
    object can also be used for other instrument types, not just tide gauges.
    For example moorings.

    Every port index for this object should use the same time coordinates.
    Therefore, timeseries need to be aligned before being placed into the
    object. If there is any padding needed, then NaNs should be used. NaNs
    should also be used for quality control/data rejection.
    """

    def init():
        return

    ##############################################################################
    ###                ~            Plotting             ~                     ###
    ##############################################################################

    ##############################################################################
    ###                ~        Model Comparison         ~                     ###
    ##############################################################################

    ##############################################################################
    ###                ~            Analysis             ~                     ###
    ##############################################################################

    def match_missing_values(self, other, fill_value=np.nan):
        """
        Will match any missing values between two tidegauge_multiple datasets.
        Where missing values (defined by fill_value) are found in either dataset
        they are also placed in the corresponding location in the other dataset.
        Returns two new tidegaugeMultiple objects containing the new
        ssh data. Datasets must contain ssh variables and only ssh will be
        masked.
        """

        ds_self = self.dataset
        ds_other = other.dataset

        ssh_self = ds_self.ssh
        ssh_other = ds_other.ssh

        if ssh_other.dims[0] == "t_dim":
            ssh_other = ssh_other.transpose()
        if ssh_self.dims[0] == "t_dim":
            ssh_self = ssh_self.transpose()

        if np.isnan(fill_value):
            ind_self = uf.isnan(ssh_self)
            ind_other = uf.isnan(ssh_other)
        else:
            ind_self = ssh_self == fill_value
            ind_other = ssh_other == fill_value

        ds_self["ssh"] = ssh_self.where(~ind_other)
        ds_other["ssh"] = ssh_other.where(~ind_self)

        tg_self = TidegaugeMultiple()
        tg_other = TidegaugeMultiple()

        tg_self.dataset = ds_self
        tg_other.dataset = ds_other

        return tg_self, tg_other

    def harmonic_analysis_utide(
        self, min_datapoints=1000, nodal=False, trend=False, method="ols", conf_int="linear", Rayleigh_min=0.95
    ):
        """
        Does a harmonic analysis for each timeseries inside this object using
        the utide library. All arguments except min_datapoints are arguments
        that are passed to ut.solve(). Please see the utide  website for more
        information:

            https://pypi.org/project/UTide/

        Utide will by default do it's harmonic analysis using a set of harmonics
        determined using the Rayleigh criterion. This changes the number of
        harmonics depending on the length and frequency of the time series.

        Output from this routine is not a new dataset, but a list of utide
        analysis object. These are structures containing, amongst other things,
        amplitudes, phases, constituent names and confidence intervals. This
        list can be passed to reconstruct_tide_utide() in this object to create
        a new TidegaugeMultiple object containing reconstructed tide data.

        INPUTS
         min_datapoints : If a time series has less than this value number of
                          datapoints, then omit from the analysis.
         <all_others>   : Inputs to utide.solve(). See website above.

        OUTPUTS
         A list of utide structures from the solve() routine. If a location
         is omitted, it will contain [] for it's entry.
        """
        ds = self.dataset
        n_port = ds.dims["id"]
        n_time = ds.dims["t_dim"]
        # Harmonic analysis datenums
        time = mdates.date2num(ds.time.values)

        analyses = []

        for pp in range(0, n_port):

            # Temporary in-loop datasets
            ds_port = ds.isel(id=pp).load()
            ssh = ds_port.ssh

            number_of_nan = np.sum(np.isnan(ssh.values))

            if number_of_nan == n_time:
                analyses.append([])
                continue

            if (n_time - number_of_nan) < min_datapoints:
                analyses.append([])
                continue

            # Do harmonic analysis using UTide
            uts_obs = ut.solve(
                time,
                ssh.values,
                lat=ssh.latitude.values,
                nodal=nodal,
                trend=trend,
                method=method,
                conf_int=conf_int,
                Rayleigh_min=Rayleigh_min,
            )

            analyses.append(uts_obs)

        return analyses

    def reconstruct_tide_utide(self, utide_solution_list, constit=None):
        """
        Use the time information inside this object to construct a tidal time
        series using a list of utide analysis objects. This list can be obtained
        using harmonic_analysis_utide(). Specify constituents to use in the
        reconstruction by passing a list of strings such as 'M2' to the constit
        argument. This won't work if a specified constituent is not present in
        the analysis.
        """

        ds = self.dataset
        n_port = ds.dims["id"]
        n_time = ds.dims["t_dim"]
        # Harmonic analysis datenums
        time = mdates.date2num(ds.time.values)

        coords = xr.Dataset(ds[list(ds.coords.keys())])
        reconstructed = np.zeros((n_port, n_time)) * np.nan

        for pp in np.arange(n_port):

            # Reconstruct full tidal signal
            pp_solution = utide_solution_list[pp]

            if len(pp_solution) == 0:
                continue

            tide = np.array(ut.reconstruct(time, pp_solution, constit=constit).h)
            reconstructed[pp] = tide

        coords["ssh_tide"] = (["id", "t_dim"], reconstructed)
        tg_return = TidegaugeMultiple()
        tg_return.dataset = coords

        return tg_return

    def calculate_residuals(self, tg_tide, apply_filter=True, window_length=25, polyorder=3):
        """
        Calculate non tidal residuals using the Total water level in THIS object
        and the tide data in another object. This assumes that this object
        contains a 'ssh' variable and the tide object contains a 'ssh_tide'
        variable. This tide variable can be constructed using e.g.
        reconstruct_tide_utide().
        """

        # NTR: Calculate non tidal residuals
        ntr = self.dataset.ssh - tg_tide.dataset.ssh_tide
        n_port = self.dataset.dims["id"]

        # NTR: Apply filter if wanted
        if apply_filter:
            for pp in range(n_port):
                ntr[pp, :] = signal.savgol_filter(ntr[pp, :], 25, 3)

        coords = xr.Dataset(self.dataset[list(self.dataset.coords.keys())])
        coords["ntr"] = ntr

        tg_return = TidegaugeMultiple()
        tg_return.dataset = coords
        return tg_return

    def threshold_statistics(self, thresholds=np.arange(-0.4, 2, 0.1), peak_separation=12):
        """
        Do some threshold statistics for all variables with a time dimension
        inside this tidegauge_multiple object. Specifically, this routine will
        calculate:

                peak_count          : The number of indepedent peaks over
                                      each specified threshold. Independent peaks
                                      are defined using the peak_separation
                                      argument. This is the number of datapoints
                                      either side of a peak within which data
                                      is ommited for further peak search.
                time_over_threshold : The total time spent over each threshold
                                      This is NOT an integral, but simple a count
                                      of all points over threshold.
                dailymax_count      : A count of the number of daily maxima over
                                      each threshold
                monthlymax_count    : A count of the number of monthly maxima
                                      over each threshold.

        Output is a xarray dataset containing analysed variables. The name of
        each analysis variable is constructed using the original variable name
        and one of the above analysis categories.
        """

        ds = self.dataset
        ds_thresh = xr.Dataset(ds[list(ds.coords.keys())])
        ds_thresh["threshold"] = ("threshold", thresholds)
        var_list = list(ds.keys())
        n_thresholds = len(thresholds)
        n_port = ds.dims["id"]

        for vv in var_list:

            empty_thresh = np.zeros((n_port, n_thresholds)) * np.nan
            ds_thresh["peak_count_" + vv] = (["id", "threshold"], np.array(empty_thresh))
            ds_thresh["time_over_threshold_" + vv] = (["id", "threshold"], np.array(empty_thresh))
            ds_thresh["dailymax_count_" + vv] = (["id", "threshold"], np.array(empty_thresh))
            ds_thresh["monthlymax_count_" + vv] = (["id", "threshold"], np.array(empty_thresh))

            for pp in range(n_port):

                # Identify NTR peaks for threshold analysis
                data_pp = ds[vv].isel(id=pp)
                if np.sum(np.isnan(data_pp.values)) == ds.dims["t_dim"]:
                    continue

                pk_ind, _ = signal.find_peaks(data_pp.values, distance=peak_separation)
                pk_values = data_pp[pk_ind]

                # Threshold Analysis
                for nn in range(0, n_thresholds):

                    # Calculate daily and monthly maxima for threshold analysis
                    ds_daily = data_pp.groupby("time.day")
                    ds_daily_max = ds_daily.max(skipna=True)
                    ds_monthly = data_pp.groupby("time.month")
                    ds_monthly_max = ds_monthly.max(skipna=True)

                    threshn = thresholds[nn]
                    # NTR: Threshold Frequency (Peaks)
                    ds_thresh["peak_count_" + vv][pp, nn] = np.sum(pk_values >= threshn)

                    # NTR: Threshold integral (Time over threshold)
                    ds_thresh["time_over_threshold_" + vv][pp, nn] = np.sum(data_pp >= threshn)

                    # NTR: Number of daily maxima over threshold
                    ds_thresh["dailymax_count_" + vv][pp, nn] = np.sum(ds_daily_max.values >= threshn)

                    # NTR: Number of monthly maxima over threshold
                    ds_thresh["monthlymax_count_" + vv][pp, nn] = np.sum(ds_monthly_max.values >= threshn)

        return ds_thresh

    def demean_timeseries(self):
        """
        Subtract time means from all variables within this tidegauge_multiple
        object. This is done independently for each id location.
        """
        return self.dataset - self.dataset.mean(dim="t_dim")

    def obs_operator(self, gridded, time_interp="nearest"):
        """
        Regrids a Gridded object onto a tidegauge_multiple object. A nearest
        neighbour interpolation is done for spatial interpolation and time
        interpolation can be specified using the time_interp argument. This
        takes any scipy interpolation string. If Gridded object contains a
        landmask variables, then the nearest WET point is taken for each tide
        gauge.

        Output is a new tidegauge_multiple object containing interpolated data.
        """

        gridded = gridded.dataset
        ds = self.dataset

        # Determine spatial indices
        print("Calculating spatial indices.", flush=True)
        ind_x, ind_y = general_utils.nearest_indices_2d(
            gridded.longitude, gridded.latitude, ds.longitude, ds.latitude, mask=gridded.landmask
        )

        # Extract spatial time series
        print("Calculating time indices.", flush=True)
        extracted = gridded.isel(x_dim=ind_x, y_dim=ind_y)
        extracted = extracted.swap_dims({"dim_0": "id"})

        # Compute data (takes a while..)
        print(" Indexing model data at tide gauge locations.. ", flush=True)
        extracted.load()

        # Check interpolation distances
        print("Calculating interpolation distances.", flush=True)
        interp_dist = general_utils.calculate_haversine_distance(
            extracted.longitude, extracted.latitude, ds.longitude.values, ds.latitude.values
        )

        # Interpolate model onto obs times
        print("Interpolating in time...", flush=True)
        extracted = extracted.rename({"time": "t_dim"})
        extracted = extracted.interp(t_dim=ds.time.values, method=time_interp)

        # Put interp_dist into dataset
        extracted["interp_dist"] = interp_dist
        extracted = extracted.rename_vars({"t_dim": "time"})

        tg_out = TidegaugeMultiple()
        tg_out.dataset = extracted
        return tg_out

    def difference(self, other, absolute_diff=True, square_diff=True):
        """
        Calculates differences between two tide gauge objects. Will calculate
        differences, absolute differences and square differences between all
        common variables within each object. Each object should have the same
        sized dimensions. When calling this routine, the differencing is done
        as follows:

            dataset1.difference(dataset2)

        This will do dataset1 - dataset2.

        Output is a new tidegauge object containing differenced variables.
        """

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

        return_differenced = TidegaugeMultiple()
        return_differenced.dataset = differenced

        return return_differenced
