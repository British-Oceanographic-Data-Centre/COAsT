import numpy as np
import xarray as xr
from . import Tidegauge, general_utils
import matplotlib.dates as mdates
import utide as ut
import scipy.signal as signal


class TidegaugeAnalysis:
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

    def __init__():
        return

    @classmethod
    def match_missing_values(cls, tidegauge1, tidegauge2, fill_value=np.nan):
        """
        Will match any missing values between two tidegauge_multiple datasets.
        Where missing values (defined by fill_value) are found in either dataset
        they are also placed in the corresponding location in the other dataset.
        Returns two new tidegaugeMultiple objects containing the new
        ssh data. Datasets must contain ssh variables and only ssh will be
        masked.
        """

        ds1 = tidegauge1.dataset
        ds2 = tidegauge2.dataset

        ssh1 = ds1.ssh
        ssh2 = ds2.ssh

        if ssh2.dims[0] == "t_dim":
            ssh2 = ssh2.transpose()
        if ssh1.dims[0] == "t_dim":
            ssh1 = ssh1.transpose()

        if np.isnan(fill_value):
            ind1 = np.isnan(ssh1)
            ind2 = np.isnan(ssh2)
        else:
            ind1 = ssh1 == fill_value
            ind2 = ssh2 == fill_value

        ds1["ssh"] = ssh1.where(~ind2)
        ds2["ssh"] = ssh2.where(~ind1)

        return Tidegauge(dataset=ds1), Tidegauge(dataset=ds2)

    @classmethod
    def harmonic_analysis_utide(cls, dataset, var_name = 'ssh', 
                                min_datapoints=1000, nodal=False, trend=False, 
                                method="ols", conf_int="linear", 
                                Rayleigh_min=0.95
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
        ds = dataset
        n_port = ds.dims["id"]
        n_time = ds.dims["t_dim"]
        # Harmonic analysis datenums
        time = mdates.date2num(ds.time.values)

        analyses = []

        for pp in range(0, n_port):

            # Temporary in-loop datasets
            ds_port = ds.isel(id=pp).load()
            var = ds_port[var_name]

            number_of_nan = np.sum(np.isnan(var.values))

            if number_of_nan == n_time:
                analyses.append([])
                continue

            if (n_time - number_of_nan) < min_datapoints:
                analyses.append([])
                continue

            # Do harmonic analysis using UTide
            uts_obs = ut.solve(
                time,
                var.values,
                lat=var.latitude.values,
                nodal=nodal,
                trend=trend,
                method=method,
                conf_int=conf_int,
                Rayleigh_min=Rayleigh_min,
            )

            analyses.append(uts_obs)

        return analyses

    def reconstruct_tide_utide(self, utide_solution_list, 
                               output_var_name = 'ssh_tide', constit=None):
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

        coords[output_var_name] = (["id", "t_dim"], reconstructed)
        tg_return = Tidegauge()
        tg_return.dataset = coords

        return tg_return

    @classmethod
    def calculate_residuals(cls, tg_ssh, tg_tide, apply_filter=True, 
                            window_length=25, polyorder=3):
        """
        Calculate non tidal residuals using the Total water level in THIS object
        and the tide data in another object. This assumes that this object
        contains a 'ssh' variable and the tide object contains a 'ssh_tide'
        variable. This tide variable can be constructed using e.g.
        reconstruct_tide_utide().
        """

        # NTR: Calculate non tidal residuals
        ntr = tg_ssh.dataset.ssh - tg_tide.dataset.ssh_tide
        n_port = tg_ssh.dataset.dims["id"]

        # NTR: Apply filter if wanted
        if apply_filter:
            for pp in range(n_port):
                ntr[pp, :] = signal.savgol_filter(ntr[pp, :], 25, 3)

        coords = xr.Dataset(tg_ssh.dataset[list(tg_ssh.dataset.coords.keys())])
        coords["ntr"] = ntr

        tg_return = Tidegauge()
        tg_return.dataset = coords
        return tg_return

    @classmethod
    def threshold_statistics(cls, dataset, thresholds=np.arange(-0.4, 2, 0.1), 
                             peak_separation=12):
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

        ds = dataset
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

    @classmethod
    def difference(cls, dataset1, dataset2, 
                   absolute_diff=True, square_diff=True):
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

        differenced = dataset1 - dataset2
        diff_vars = list(differenced.keys())
        save_coords = list(dataset1.coords.keys())

        for vv in diff_vars:
            differenced = differenced.rename({vv: "diff_" + vv})

        if absolute_diff:
            abs_tmp = np.fabs(differenced)
            diff_vars = list(abs_tmp.keys())
            for vv in diff_vars:
                abs_tmp = abs_tmp.rename({vv: "abs_" + vv})
        else:
            abs_tmp = xr.Dataset()

        if square_diff:
            sq_tmp = np.square(differenced)
            diff_vars = list(sq_tmp.keys())
            for vv in diff_vars:
                sq_tmp = sq_tmp.rename({vv: "square_" + vv})
        else:
            sq_tmp = xr.Dataset()

        differenced = xr.merge((differenced, abs_tmp, sq_tmp, 
                                dataset1[save_coords]))

        return_differenced = Tidegauge()
        return_differenced.dataset = differenced

        return return_differenced
