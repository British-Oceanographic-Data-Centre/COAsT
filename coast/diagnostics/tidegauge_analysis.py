"""An analysis class for tide gauge."""
import numpy as np
import xarray as xr
from ..data.tidegauge import Tidegauge
from .._utils import general_utils, stats_util, crps_util
import matplotlib.dates as mdates
import utide as ut
import scipy.signal as signal


class TidegaugeAnalysis:
    """
    This contains analysis methods suitable for use with the dataset structure
    of Tidegauge()
    """

    def __init__(self):
        return

    @classmethod
    def match_missing_values(cls, data_array1, data_array2, fill_value=np.nan):
        """
        Will match any missing values between two tidegauge_multiple datasets.
        Where missing values (defined by fill_value) are found in either dataset
        they are also placed in the corresponding location in the other dataset.
        Returns two new tidegaugeMultiple objects containing the new
        ssh data. Datasets must contain ssh variables and only ssh will be
        masked.
        """

        if data_array2.dims[0] == "t_dim":
            data_array2 = data_array2.transpose()
        if data_array1.dims[0] == "t_dim":
            data_array1 = data_array1.transpose()

        if np.isnan(fill_value):
            ind1 = np.isnan(data_array1.values)
            ind2 = np.isnan(data_array2.values)
        else:
            ind1 = data_array1.values == fill_value
            ind2 = data_array2.values == fill_value

        ds1 = data_array1.where(~ind2)
        ds2 = data_array2.where(~ind1)

        return Tidegauge(dataset=ds1.to_dataset()), Tidegauge(dataset=ds2.to_dataset())

    @classmethod
    def harmonic_analysis_utide(
        cls,
        data_array,
        min_datapoints=1000,
        nodal=False,
        trend=False,
        method="ols",
        conf_int="linear",
        Rayleigh_min=0.95,
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
         data_array     : Xarray data_array from a coast.Tidegauge() object
                          e.g. tidegauge.dataset.ssh
         min_datapoints : If a time series has less than this value number of
                          datapoints, then omit from the analysis.
         <all_others>   : Inputs to utide.solve(). See website above.

        OUTPUTS
         A list of utide structures from the solve() routine. If a location
         is omitted, it will contain [] for it's entry.
        """
        # Make name shorter for computations and get dimension lengths
        ds = data_array
        n_port = ds.sizes["id_dim"]
        n_time = ds.sizes["t_dim"]

        # Harmonic analysis datenums -- for utide to work correctly < 0.3.0
        # time = mdates.date2num(ds.time.values)
        time = ds.time.values  # accepts np.datetime64 uTide >= 0.3.0

        # Create empty list of analyses
        analyses = []

        # Loop over ports
        for pp in range(0, n_port):

            # Temporary in-loop datasets
            ds_port = ds.isel(id_dim=pp).load()
            number_of_nan = np.sum(np.isnan(ds_port.values))

            # If not enough datapoints for analysis then append an empty list
            if (n_time - number_of_nan) < min_datapoints:
                analyses.append([])
                continue

            # Do harmonic analysis using UTide
            uts_obs = ut.solve(
                time,
                ds_port.values,
                lat=ds_port.latitude.values,
                nodal=nodal,
                trend=trend,
                method=method,
                conf_int=conf_int,
                Rayleigh_min=Rayleigh_min,
            )

            analyses.append(uts_obs)

        return analyses

    @classmethod
    def reconstruct_tide_utide(cls, data_array, utide_solution_list, constit=None, output_name="reconstructed"):
        """
        Use the tarray of times to reconstruct a time series series using a
        list of utide analysis objects. This list can be obtained
        using harmonic_analysis_utide(). Specify constituents to use in the
        reconstruction by passing a list of strings such as 'M2' to the constit
        argument. This won't work if a specified constituent is not present in
        the analysis.
        """

        # Get dimension lengths
        n_port = len(utide_solution_list)
        n_time = len(data_array.time)

        # Harmonic analysis datenums -- needed for utide < 0.3.0
        # time = mdates.date2num(data_array.time)
        time = data_array.time  # accepts np.datetime64 uTide >= 0.3.0

        # Get  coordinates from data_array and convert to Dataset for output
        reconstructed = np.zeros((n_port, n_time)) * np.nan

        # Loop over ports
        for pp in np.arange(n_port):

            # Reconstruct full tidal signal using utide
            pp_solution = utide_solution_list[pp]
            if len(pp_solution) == 0:
                continue

            # Call utide.reconstruct
            tide = np.array(ut.reconstruct(time, pp_solution, constit=constit).h)
            reconstructed[pp] = tide

        # Create output dataset and return it in new Tidegauge object.
        ds_out = xr.Dataset(data_array.coords)
        ds_out[output_name] = (["id_dim", "t_dim"], reconstructed)

        return Tidegauge(dataset=ds_out)

    @classmethod
    def calculate_non_tidal_residuals(
        cls, data_array_ssh, data_array_tide, apply_filter=True, window_length=25, polyorder=3
    ):
        """
        Calculate non tidal residuals by subtracting values in data_array_tide
        from data_array_ssh. You may optionally apply a filter to the non
        tidal residual data by setting apply_filter = True. This uses the
        scipy.signal.savgol_filter function, which you ay pass window_length
        and poly_order.
        """

        # NTR: Calculate non tidal residuals
        ntr = data_array_ssh - data_array_tide
        n_port = data_array_ssh.sizes["id_dim"]

        # NTR: Apply filter if wanted
        if apply_filter:
            for pp in range(n_port):
                ntr[pp, :] = signal.savgol_filter(ntr[pp, :], 25, 3)

        # Create output Tidegauge object and return
        ds_coords = data_array_ssh.coords.to_dataset()
        ds_coords["ntr"] = ntr
        return Tidegauge(dataset=ds_coords)

    @classmethod
    def threshold_statistics(cls, dataset, thresholds=np.arange(-0.4, 2, 0.1), peak_separation=12):
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

        # Set up working datasets and lists
        ds = dataset
        ds_thresh = xr.Dataset(ds.coords)
        ds_thresh["threshold"] = ("threshold", thresholds)
        var_list = list(ds.keys())
        n_thresholds = len(thresholds)
        n_port = ds.sizes["id_dim"]

        # Loop over vars in the input dataset
        for vv in var_list:

            empty_thresh = np.zeros((n_port, n_thresholds)) * np.nan
            ds_thresh["peak_count_" + vv] = (["id_dim", "threshold"], np.array(empty_thresh))
            ds_thresh["time_over_threshold_" + vv] = (["id_dim", "threshold"], np.array(empty_thresh))
            ds_thresh["dailymax_count_" + vv] = (["id_dim", "threshold"], np.array(empty_thresh))
            ds_thresh["monthlymax_count_" + vv] = (["id_dim", "threshold"], np.array(empty_thresh))

            for pp in range(n_port):

                # Identify NTR peaks for threshold analysis
                data_pp = ds[vv].isel(id_dim=pp)
                if np.sum(np.isnan(data_pp.values)) == ds.sizes["t_dim"]:
                    continue

                pk_ind, _ = signal.find_peaks(data_pp.values.copy(), distance=peak_separation)
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

    @staticmethod
    def demean_timeseries(dataset):
        """
        Subtract time means from all variables within this tidegauge_multiple
        object. This is done independently for each id_dim location.
        """
        demeaned = dataset - dataset.mean(dim="t_dim")
        return Tidegauge(dataset=demeaned)

    @classmethod
    def difference(cls, dataset1, dataset2, absolute_diff=True, square_diff=True):
        """
        Calculates differences between two tide gauge objects datasets. Will calculate
        differences, absolute differences and square differences between all
        common variables within each object. Each object should have the same
        sized dimensions. When calling this routine, the differencing is done
        as follows:

            dataset1.difference(dataset2)

        This will do dataset1 - dataset2.

        Output is a new tidegauge object containing differenced variables.
        """

        # Get all differences and save coordintes for later
        differenced = dataset1 - dataset2
        diff_vars = list(differenced.keys())
        save_coords = list(dataset1.coords.keys())

        # Loop oer all variables
        for vv in diff_vars:
            differenced = differenced.rename({vv: "diff_" + vv})

        # Calculate absolute differences maybe
        if absolute_diff:
            abs_tmp = np.fabs(differenced)
            diff_vars = list(abs_tmp.keys())
            for vv in diff_vars:
                abs_tmp = abs_tmp.rename({vv: "abs_" + vv})
        else:
            abs_tmp = xr.Dataset()

        # Calculate squared differences maybe
        if square_diff:
            sq_tmp = np.square(differenced)
            diff_vars = list(sq_tmp.keys())
            for vv in diff_vars:
                sq_tmp = sq_tmp.rename({vv: "square_" + vv})
        else:
            sq_tmp = xr.Dataset()

        # Merge all differences into one
        differenced = xr.merge((differenced, abs_tmp, sq_tmp, dataset1[save_coords]))

        return Tidegauge(dataset=differenced)

    @staticmethod
    def find_high_and_low_water(data_array, method="comp", **kwargs):
        """
        Finds high and low water for a given variable.
        Returns in a new TIDEGAUGE object with similar data format to
        a TIDETABLE. If this Tidegauge object contains more than one location
        (id_dim > 1) then a list of Tidegauges will be returned.

        Methods:
        'comp' :: Find maxima by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        'cubic':: Find the maxima using the roots of cubic spline.
                  Uses scipy.interpolate.InterpolatedUnivariateSpline
                  and scipy.signal.argrelmax. **kwargs are not activated.
        NOTE: Currently only the 'comp' and 'cubic' methods implemented. Future
                  methods include linear interpolation or refinements.
        """

        if "id_dim" in data_array.dims:
            n_id = data_array.sizes["id_dim"]
        else:
            n_id = 1
            data_array = data_array.expand_dims("id_dim")

        tg_list = []
        # Loop over id_dim dimension
        for ii in range(n_id):
            # Get x and y for input to find_maxima

            x = data_array.isel(id_dim=ii).time.values
            y = data_array.isel(id_dim=ii).values

            # Get time and values of maxima and minima
            time_max, values_max = stats_util.find_maxima(x, y, method=method, **kwargs)
            time_min, values_min = stats_util.find_maxima(x, -y, method=method, **kwargs)
            # Place the above values into a brand new dataset for this id_dim index
            new_dataset = xr.Dataset()
            new_dataset.attrs = data_array.attrs
            new_dataset[data_array.name + "_highs"] = ("time_highs", values_max)
            new_dataset[data_array.name + "_lows"] = ("time_lows", -values_min)
            new_dataset["time_highs"] = ("time_highs", time_max)
            new_dataset["time_lows"] = ("time_lows", time_min)

            # Place dataset into a new Tidegauge object and append to output
            new_object = Tidegauge(dataset=new_dataset)
            tg_list.append(new_object)

        # If only 1 index, return just a Tidegauge object, else return list.
        if n_id == 1:
            return new_object
        else:
            return tg_list

    @staticmethod
    def doodson_x0_filter(dataset, var_str):
        """Applies doodson X0 filter to a specified TIDEGAUGE variable
        Input ius expected to be hourly. Use resample_mean to average data
        to hourly frequency."""
        n_id = dataset.sizes["id_dim"]
        n_time = dataset.sizes["t_dim"]
        filtered = np.zeros((n_id, n_time))
        for ii in range(n_id):
            ds_ii = dataset.isel(id_dim=ii)
            filtered[ii] = stats_util.doodson_x0_filter(ds_ii[var_str], ax=0)
        new_dataset = xr.Dataset(dataset.coords)
        new_dataset[var_str] = (("id_dim", "t_dim"), filtered)
        return Tidegauge(dataset=new_dataset)

    @classmethod
    def crps(
        cls,
        tidegauge_data,
        gridded_data,
        nh_radius: float = 20,
        time_interp: str = "linear",
    ):
        """
        Comparison of observed variable to modelled using the Continuous
        Ranked Probability Score. This is done using this TIDEGAUGE object.
        This method specifically performs a single-observation neighbourhood-
        forecast method.

        Parameters
        ----------
        model_object (model) : Model object (NEMO) containing model data
        model_var_name (str) : Name of model variable to compare.
        obs_var_name (str)   : Name of observed variable to compare.
        nh_radius (float)    : Neighbourhood rad
        cdf_type (str)       : Type of cumulative distribution to use for the
                               model data ('empirical' or 'theoretical').
                               Observations always use empirical.
        time_interp (str)    : Type of time interpolation to use (s)
        create_new_obj (bool): If True, save output to new TIDEGAUGE obj.
                               Otherwise, save to this obj.

        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags

        Example Useage
        -------
        # Compare modelled 'sossheig' with 'ssh' using CRPS
        crps = altimetry.crps(nemo, 'sossheig', 'ssh')
        """

        # Extract variables and get shape
        mod_var = gridded_data
        obs_var = tidegauge_data
        n_id, n_time = obs_var.shape

        # Make output CRPS array
        crps_out = np.zeros((n_id, n_time))
        N_out = np.zeros((n_id, n_time))

        # Loop over location indices
        for ii in range(n_id):

            obs_ii = obs_var.isel(id_dim=ii)

            # Calculate CRPS
            crps_list, n_model_pts, contains_land = crps_util.crps_sonf_fixed(
                mod_var,
                tidegauge_data.longitude.values,
                tidegauge_data.latitude.values,
                obs_ii.values,
                tidegauge_data.time.values,
                nh_radius,
                time_interp,
            )

            crps_out[ii] = crps_list
            N_out[ii] = n_model_pts

        # Put into new object
        new_dataset = xr.Dataset(tidegauge_data.coords)
        new_dataset["crps"] = (("id_dim", "t_dim"), crps_out)
        new_dataset["crps_N"] = (("id_dim", "t_dim"), N_out)
        return Tidegauge(dataset=new_dataset)

    @classmethod
    def time_mean(cls, dataset, date0=None, date1=None):
        """Time mean of all variables between dates date0, date1"""
        var = general_utils.data_array_time_slice(dataset, date0, date1)
        return Tidegauge(dataset=var.mean(dim="t_dim", skipna=True))

    @classmethod
    def time_std(cls, dataset, date0=None, date1=None):
        """Time st. dev of variable var_str between dates date0 and date1"""
        var = general_utils.data_array_time_slice(dataset, date0, date1)
        return Tidegauge(dataset=var.std(dim="t_dim", skipna=True))

    @classmethod
    def time_slice(cls, dataset, date0=None, date1=None):
        sliced = general_utils.data_array_time_slice(dataset, date0, date1)
        return Tidegauge(dataset=sliced)

    @classmethod
    def resample_mean(cls, dataset, time_freq: str, **kwargs):
        """Resample a TIDEGAUGE variable in time by calculating the mean
            of all data points at a given frequency.

        Parameters
        ----------
        time_freq (str)  : Time frequency. e.g. '1H' for hourly, '1D' for daily
                           Can also be a timedelta object. See Pandas resample
                           method for more info.
        **kwargs (other) : Other arguments to pass to xarray.Dataset.resample
        (http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html)

        Returns
        -------
        New Tidegauge() object containing resampled data
        """

        # Resample using xarray.resample
        dataset = dataset.swap_dims({"t_dim": "time"})
        resampled = dataset.resample(time=time_freq, **kwargs).mean()
        resampled = resampled.swap_dims({"time": "t_dim"})
        return Tidegauge(dataset=resampled)
