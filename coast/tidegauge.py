"""Tide Gauge class"""
from .timeseries import Timeseries
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
from typing import Union
from pathlib import Path
import xarray.ufuncs as uf


class Tidegauge(Timeseries):
    """
    An object for reading, storing and manipulating tide gauge data.
    Functionality available for reading and organisation of GESLA files.
    (Source: https://www.gesla.org/).  However, any fixed time series data can
    be used if in the correct format.

    The data format used for this object is as follows:

    *Data Format Overview*

        1. Data for a single tide gauge is stored in an xarray Dataset object.
           This can be accessed using TIDEGAUGE.dataset.
        2. The dataset has a single dimension: time.
        3. Latitude/Longitude and other single values parameters are stored as
           attributes or single float variables.
        4. Time is a coordinate variable and time dimension.
        5. Data variables are stored along the time dimension.
        6. The attributes: site_name, latitude, longitude are expected. If they
            are missing functionality may be reduced.


    *Methods Overview*

        *Initialisation and File Reading*
        -> __init__: Can be initialised with a GESLA file or empty.
        -> obs_operator: Interpolates model data to time series locations
           and times (not yet implemented).
        -> read_gesla_v3: Reads a format version 3.0
           GESLA file to an xarray Dataset.
        -> read_gesla_header_v3: Reads the header of a version 3
           GESLA file.
        -> read_gesla_data_v3: Reads data from a version 3 GESLA
           file.
        -> create_multiple_tidegauge: Creates multiple tide gauge objects
           objects from a list of filenames or directory and returns them
           in a list.

        *Plotting*
        -> plot_on_map: Plots location of TIDEGAUGE object on map.
        -> plot_timeseries: Plots a specified time series.

        *Model Comparison*
        -> obs_operator(): For interpolating model data to this object.
        -> cprs(): Calculates the CRPS between a model and obs variable.
        -> difference(): Differences two specified variables
        -> absolute_error(): Absolute difference, two variables
        -> mean_absolute_error(): MAE between two variables
        -> root_mean_square_error(): RMSE between two variables
        -> time_mean(): Mean of a variable in time
        -> time_std(): St. Dev of a variable in time
        -> time_correlation(): Correlation between two variables
        -> time_covariance(): Covariance between two variables
        -> basic_stats(): Calculates multiple of the above metrics.

        *Analysis*
        -> resample_mean(): For resampling data in time using averaging
        -> apply_doodson_xo_filter(): Remove tidal signal using Doodson XO
        -> find_high_and_low_water(): Find maxima and minima of time series
    """

    def __init__(self, dataset = None, config: Union[Path, str] = None):
        """
        Initialise TIDEGAUGE object either as empty (no arguments) or by
        reading GESLA data from a directory between two datetime objects.

        Example usage:
        --------------
        # Read tide gauge data for data in January 1990
        date0 = datetime.datetime(1990,1,1)
        date1 = datetime.datetime(1990,2,1)
        tg = coast.TIDEGAUGE(<'path_to_file'>, date0, date1)

        # Access the data
        tg.dataset

        Parameters
        ----------
        file_path (list of str) : Filename to read from directory.
        date_start (datetime) : Start date for data read. Optional
        date_end (datetime) : end date for data read. Optional
        config (Path or str) : configuration file

        Returns
        -------
        Self
        """
        debug(f"Creating a new ..... {get_slug(self)}")
        super().__init__(config)

        # If file list is supplied, read files from directory
        if dataset is not None:
            self.dataset = dataset
            self.apply_config_mappings()
        else:
            self.dataset = None
            

        print(f"{get_slug(self)} initialised")

    ############ tide gauge methods ###########################################
    def read_gesla_v3(self, fn_gesla, date_start=None, date_end=None):
        """
        For reading from a GESLA2 (Format version 3.0) file(s) into an
        xarray dataset. Formatting according to Woodworth et al. (2017).
        Website: https://www.gesla.org/
        If no data lies between the specified dates, a dataset is still created
        containing information on the tide gauge, but the time dimension will
        be empty.
        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file, list of files or a glob
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        Creates xarray.dataset within tidegauge object containing loaded data.
        If multiple files are provided then instead returns a list of NEW
        tidegauge objects.
        """
        debug(f'Reading "{fn_gesla}" as a GESLA file with {get_slug(self)}')  
        # TODO Maybe include start/end dates

        # See if its a file list input, or a glob
        if type(fn_gesla) is not list:
            file_list = glob.glob(fn_gesla)
            
        multiple=False
        if len(file_list) > 1:
            multiple=True

        ds_list = []
        # Loop over files and put resulting dataset into an output list
        for fn in file_list:
            try:
                header_dict = self._read_gesla_header_v3(fn)
                dataset = self._read_gesla_data_v3(fn, date_start, date_end)
            except:
                raise Exception("Problem reading GESLA file: " + fn)
            # Attributes
            dataset["longitude"] = header_dict["longitude"]
            dataset["latitude"] = header_dict["latitude"]
            del header_dict["longitude"]
            del header_dict["latitude"]
            dataset.attrs = header_dict
            
            # Create tidegauge object, save dataset and append to list
            if multiple:
                tg_tmp = Tidegauge()
                tg_tmp.dataset = dataset
                tg_tmp.apply_config_mappings()
                ds_list.append(tg_tmp)

        # If there is only one file, then just return the dataset, not a list
        if multiple:
            return ds_list
        else:
            self.dataset = dataset
            self.apply_config_mappings()

    @classmethod
    def _read_gesla_header_v3(cls, fn_gesla):
        """
        Reads header from a GESLA file (format version 3.0).

        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file

        Returns
        -------
        dictionary of attributes
        """
        debug(f'Reading GESLA header from "{fn_gesla}"')
        fid = open(fn_gesla)

        # Read lines one by one (hopefully formatting is consistent)
        fid.readline()  # Skip first line
        # Geographical stuff
        site_name = fid.readline().split()[3:]
        site_name = "_".join(site_name)
        country = fid.readline().split()[2:]
        country = "_".join(country)
        contributor = fid.readline().split()[2:]
        contributor = "_".join(contributor)
        # Coordinates
        latitude = float(fid.readline().split()[2])
        longitude = float(fid.readline().split()[2])
        coordinate_system = fid.readline().split()[3]
        # Dates
        start_date = fid.readline().split()[3:5]
        start_date = " ".join(start_date)
        start_date = pd.to_datetime(start_date)
        end_date = fid.readline().split()[3:5]
        end_date = " ".join(end_date)
        end_date = pd.to_datetime(end_date)
        time_zone_hours = float(fid.readline().split()[4])
        # Other
        fid.readline()  # Datum
        fid.readline()  # Instrument
        precision = float(fid.readline().split()[2])
        null_value = float(fid.readline().split()[3])

        debug(f'Read done, close file "{fn_gesla}"')
        fid.close()
        # Put all header info into an attributes dictionary
        header_dict = {
            "site_name": site_name,
            "country": country,
            "contributor": contributor,
            "latitude": latitude,
            "longitude": longitude,
            "coordinate_system": coordinate_system,
            "original_start_date": start_date,
            "original_end_date": end_date,
            "time_zone_hours": time_zone_hours,
            "precision": precision,
            "null_value": null_value,
        }
        return header_dict

    @classmethod
    def _read_gesla_data_v3(cls, fn_gesla, date_start=None, 
                            date_end=None, header_length: int = 32):
        """
        Reads observation data from a GESLA file (format version 3.0).

        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags
        """
        # Initialise empty dataset and lists
        debug(f'Reading GESLA data from "{fn_gesla}"')
        dataset = xr.Dataset()
        time = []
        ssh = []
        qc_flags = []
        # Open file and loop until EOF
        with open(fn_gesla) as file:
            line_count = 1
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count > header_length:
                    working_line = line.split()
                    if working_line[0] != "#":
                        time.append(working_line[0] + " " + working_line[1])
                        ssh.append(float(working_line[2]))
                        qc_flags.append(int(working_line[3]))

                line_count = line_count + 1
            debug(f'Read done, close file "{fn_gesla}"')

        # Convert time list to datetimes using pandas
        time = np.array(pd.to_datetime(time))

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            date_start = np.datetime64(date_start)
            start_index = np.argmax(time >= date_start)
        if date_end is not None:
            date_end = np.datetime64(date_end)
            end_index = np.argmax(time > date_end)
        time = time[start_index:end_index]
        ssh = ssh[start_index:end_index]
        qc_flags = qc_flags[start_index:end_index]

        # Set null values to nan
        ssh = np.array(ssh)
        qc_flags = np.array(qc_flags)
        ssh[qc_flags == 5] = np.nan

        # Assign arrays to Dataset
        dataset["ssh"] = xr.DataArray(ssh, dims=["t_dim"]).expand_dims("id")
        dataset["qc_flags"] = xr.DataArray(qc_flags, dims=["t_dim"]).expand_dims("id")
        dataset = dataset.assign_coords(time=("t_dim", time))

        # Assign local dataset to object-scope dataset
        return dataset

    ### tide table methods (HLW) 
    def read_hlw(self, fn_hlw, date_start=None, date_end=None):
        """
        For reading from a file of tidetable High and Low Waters (HLW) data into an
        xarray dataset. File contains high water and low water heights and times

        If no data lies between the specified dates, a dataset is still created
        containing information on the tide gauge, but the time dimension will
        be empty.

        The data takes the form:
        LIVERPOOL (GLADSTONE DOCK)    TZ: UT(GMT)/BST     Units: METRES    Datum: Chart Datum
        01/10/2020  06:29    1.65
        01/10/2020  11:54    9.01
        01/10/2020  18:36    1.87
        ...

        Parameters
        ----------
        fn_hlw (str) : path to tabulated High Low Water file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        """
        debug(f'Reading "{fn_hlw}" as a HLW file with {get_slug(self)}')  
        # TODO Maybe include start/end dates
        try:
            header_dict = self._read_hlw_header(fn_hlw)
            dataset = self._read_hlw_data(fn_hlw, header_dict, date_start, date_end)
            if header_dict["field"] == "TZ:UT(GMT)/BST":
                debug("Read in as BST, stored as UTC")
            elif header_dict["field"] == "TZ:GMTonly":
                debug("Read and store as GMT/UTC")
            else:
                debug("Not expecting that timezone")

        except:
            raise Exception("Problem reading HLW file: " + fn_hlw)

        dataset.attrs = header_dict
        self.dataset = dataset
        self.apply_config_mappings()

    @classmethod
    def _read_hlw_header(cls, filnam):
        """
        Reads header from a HWL file.

        The data takes the form:
        LIVERPOOL (GLADSTONE DOCK) TZ: UT(GMT)/BST Units: METRES Datum: Chart Datum
        01/10/2020  06:29    1.65
        01/10/2020  11:54    9.01
        01/10/2020  18:36    1.87
        ...

        Parameters
        ----------
        filnam (str) : path to file

        Returns
        -------
        dictionary of attributes
        """
        debug(f'Reading HLW header from "{filnam}" ')
        fid = open(filnam)

        # Read lines one by one (hopefully formatting is consistent)
        header = re.split(r"\s{2,}", fid.readline())
        site_name = header[0]
        site_name = site_name.replace(" ", "")

        field = header[1]
        field = field.replace(" ", "")

        units = header[2]
        units = units.replace(" ", "")

        datum = header[3]
        datum = datum.replace(" ", "")

        debug(f'Read done, close file "{filnam}"')
        fid.close()
        # Put all header info into an attributes dictionary
        header_dict = {"site_name": site_name, "field": field, 
                       "units": units, "datum": datum}
        return header_dict

    @classmethod
    def _read_hlw_data(cls, filnam, header_dict, date_start=None, 
                       date_end=None, header_length: int = 1):
        """
        Reads HLW data from a tidetable file.

        Parameters
        ----------
        filnam (str) : path to HLW tide gauge file
        date_start (np.datetime64) : start date for returning data.
        date_end (np.datetime64) : end date for returning data.
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, High and Low water values
        """
        import datetime

        # Initialise empty dataset and lists
        debug(f'Reading HLW data from "{filnam}"')
        dataset = xr.Dataset()
        time = []
        ssh = []

        if header_dict["field"] == "TZ:UT(GMT)/BST":
            localtime_flag = True
        else:
            localtime_flag = False

        # Open file and loop until EOF
        with open(filnam) as file:
            line_count = 1
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count > header_length:
                    working_line = line.split()
                    if working_line[0] != "#":
                        time_str = working_line[0] + " " + working_line[1]
                        # Read time as datetime.datetime because it can handle local timezone easily AND the unusual date format
                        datetime_obj = datetime.datetime.strptime(time_str, "%d/%m/%Y %H:%M")
                        if localtime_flag == True:
                            time.append(np.datetime64(datetime_obj.astimezone()))
                        else:
                            time.append(np.datetime64(datetime_obj))
                        ssh.append(float(working_line[2]))
                line_count = line_count + 1
            debug(f'Read done, close file "{filnam}"')

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            # start_index = general_utils.nearest_datetime_ind(time, date_start)
            date_start = np.datetime64(date_start)
            start_index = np.argmax(time >= date_start)
            debug(f"date_start: {date_start}. start_index: {start_index}")
        if date_end is not None:
            # end_index = general_utils.nearest_datetime_ind(time, date_end)
            date_end = np.datetime64(date_end)
            end_index = np.argmax(time > date_end)
            debug(f"date_end: {date_end}. end_index: {end_index}")
        time = time[start_index:end_index]
        ssh = ssh[start_index:end_index]
        debug(f"ssh: {ssh}")
        # Assign arrays to Dataset
        dataset["ssh"] = xr.DataArray(ssh, dims=["time"]).expand_dims("id")
        dataset = dataset.assign_coords(time=("time", time))
        # Assign local dataset to object-scope dataset
        return dataset

    def show(self, timezone: str = None):
        """
        Print out the values in the xarray
        Displays with specified timezone
        """
        # debug(" Saltney pred", np.datetime_as_string(Saltney_time_pred[i], unit='m', timezone=pytz.timezone('Europe/London')),". Height: {:.2f} m".format( HT.values[i] ))
        if timezone == None:
            for i in range(len(self.dataset.ssh)):
                #               debug('time:', self.dataset.time[i].values,
                debug(
                    "time (UTC):",
                    general_utils.dayoweek(self.dataset.time[i].values),
                    np.datetime_as_string(self.dataset.time[i], unit="m"),
                    "height:",
                    self.dataset.ssh[i].values,
                    "m",
                )
        else:  # display timezone aware times
            for i in range(len(self.dataset.ssh)):
                #               debug('time:', self.dataset.time[i].values,
                debug(
                    "time (" + timezone + "):",
                    general_utils.day_of_week(self.dataset.time[i].values),
                    np.datetime_as_string(self.dataset.time[i], unit="m", 
                                          timezone=pytz.timezone(timezone)),
                    "height:",
                    self.dataset.ssh[i].values,
                    "m",
                )

    def get_tide_table_times(
        self,
        time_guess: np.datetime64 = None,
        time_var: str = "time",
        measure_var: str = "ssh",
        method: str = "window",
        winsize=None,
    ):
        """
        Get tide times and heights from tide table.
        input:
        time_guess : np.datetime64 or datetime
                assumes utc
        time_var : name of time variable [default: 'time']
        measure_var : name of ssh variable [default: 'ssh']

        method =
            window:  +/- hours window size, winsize, (int) return values in that window
                uses additional variable winsize (int) [default 2hrs]
            nearest_1: return only the nearest event, if in winsize [default:None]
            nearest_2: return nearest event in future and the nearest in the past (i.e. high and a low), if in winsize [default:None]
            nearest_HW: return nearest High Water event (computed as the max of `nearest_2`), if in winsize [default:None]

        returns: xr.DataArray( measure_var, coords=time_var)
            E.g. ssh (m), time (utc)
            If value is not found, it returns a NaN with time value as the
            guess value.

        """
        # Ensure the date objects are datetime
        if type(time_guess) is not np.datetime64:
            debug("Convert date to np.datetime64")
            time_guess = np.datetime64(time_guess)

        if time_guess == None:
            debug("Use today's date")
            time_guess = np.datetime64("now")

        if method == "window":
            if winsize == None:
                winsize = 2
            # initialise start_index and end_index
            start_index = 0
            end_index = len(self.dataset[time_var])

            date_start = time_guess - np.timedelta64(winsize, "h")
            start_index = np.argmax(self.dataset[time_var].values >= date_start)

            date_end = time_guess + np.timedelta64(winsize, "h")
            end_index = np.argmax(self.dataset[time_var].values > date_end)

            ssh = self.dataset[measure_var].isel(time=slice(start_index,end_index))

            return ssh[0]

        elif method == "nearest_1":
            dt = np.abs(self.dataset[time_var] - time_guess)
            index = np.argsort(dt).values
            if winsize is not None:  # if search window trucation exists
                if np.timedelta64(dt[index[0]].values, "m").astype("int") <= 60 * winsize:  # compare in minutes
                    debug(f"dt:{np.timedelta64(dt[index[0]].values, 'm').astype('int')}")
                    debug(f"winsize:{winsize}")
                    return self.dataset[measure_var].isel(time=index[0])
                else:
                    # return a NaN in an xr.Dataset
                    # The rather odd trailing zero is to remove the array layer
                    # on both time and measurement, and to match the other
                    # alternative for a return object
                    return xr.DataArray([np.NaN], dims=(time_var), coords={time_var: [time_guess]})[0]
            else:  # give the closest without window search truncation
                return self.dataset[measure_var].isel(time=index[0])

        elif method == "nearest_2":
            index = np.argsort(np.abs(self.dataset[time_var] - time_guess)).values
            nearest_2 = self.dataset[measure_var].isel(time=index[0 : 1 + 1])  # , self.dataset.time[index[0:1+1]]
            return nearest_2[0]

        elif method == "nearest_HW":
            index = np.argsort(np.abs(self.dataset[time_var] - time_guess)).values
            # return self.dataset.ssh[ index[np.argmax( self.dataset.ssh[index[0:1+1]]] )] #, self.dataset.time[index[0:1+1]]
            nearest_2 = self.dataset[measure_var].isel(time=index[0 : 1 + 1])  # , self.dataset.time[index[0:1+1]]
            return nearest_2.isel(time=nearest_2.argmax())

        else:
            debug("Not expecting that option / method")

    ############ environment.data.gov.uk gauge methods ###########################
    @classmethod
    def read_ea_api_to_xarray(
        cls, n_days: int = 5, date_start: np.datetime64 = None, date_end: np.datetime64 = None, station_id="E70124"
    ):
        """
        load gauge data via environment.data.gov.uk EA API
        Either loads last n_days, or from date_start:date_end

        API Source:
        https://environment.data.gov.uk/flood-monitoring/doc/reference

        Details of available tidal stations are recovered with:
        https://environment.data.gov.uk/flood-monitoring/id/stations?type=TideGauge
        Recover the "stationReference" for the gauge of interest and pass as
        station_id:str. The default station_id="E70124" is Liverpool.

        INPUTS:
            n_days : int. Extact the last n_days from now.
            date_start : datetime. UTC format string "yyyy-MM-dd" E.g 2020-01-05
            date_end : datetime
            station_id : int. Station id. Also referred to as stationReference in
             EA API. Default value is for Liverpool.
        OUTPUT:
            ssh, time : xr.Dataset
        """
        import requests, json

        cls.n_days = n_days
        cls.date_start = date_start
        cls.date_end = date_end
        cls.station_id = station_id  # EA id: stationReference

        # Obtain and process header information
        info("load station info")
        url = "https://environment.data.gov.uk/flood-monitoring/id/stations/" + cls.station_id + ".json"
        try:
            request_raw = requests.get(url)
            header_dict = json.loads(request_raw.content)
        except ValueError:
            debug(f"Failed request for station {cls.station_id}")
            return

        try:
            header_dict["site_name"] = header_dict["items"]["label"]
            header_dict["latitude"] = header_dict["items"]["lat"]
            header_dict["longitude"] = header_dict["items"]["long"]
        except:
            info(f"possible missing some header info: site_name,latitude,longitude")
        try:
            # Define url call with parameter from station info
            htmlcall_station_id = header_dict["items"]["measures"]["@id"] + "/readings?"
        except:
            debug(f"problem defining the parameter to read")

        # Construct API request for data recovery
        info("load station data")
        if (cls.date_start == None) & (cls.date_end == None):
            info(f"GETting n_days= {cls.n_days} of data")
            url = (
                htmlcall_station_id
                + "since="
                + (np.datetime64("now") - np.timedelta64(n_days, "D")).item().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            debug(f"url request: {url}")
        else:
            # Check date_start and date_end are timetime objects
            if (type(cls.date_start) is np.datetime64) & (type(cls.date_end) is np.datetime64):
                info(f"GETting data from {cls.date_start} to {cls.date_end}")
                startdate = cls.date_start.item().strftime("%Y-%m-%d")
                enddate = cls.date_end.item().strftime("%Y-%m-%d")
                url = htmlcall_station_id + "startdate=" + startdate + "&enddate=" + enddate
                debug(f"url request: {url}")

            else:
                debug("Expecting date_start and date_end as datetime objects")

        # Get the data
        try:
            request_raw = requests.get(url)
            request = json.loads(request_raw.content)
            debug(f"EA API request: {request_raw.text}")
        except ValueError:
            debug(f"Failed request: {request_raw}")
            return

        # Process timeseries data
        dataset = xr.Dataset()
        time = []
        ssh = []
        nvals = len(request["items"])
        time = np.array([np.datetime64(request["items"][i]["dateTime"]) for i in range(nvals)])
        ssh = np.array([request["items"][i]["value"] for i in range(nvals)])

        # Assign arrays to Dataset
        dataset["ssh"] = xr.DataArray(ssh, dims=["time"])
        dataset = dataset.assign_coords(time=("time", time))
        dataset.attrs = header_dict
        debug(f"EA API request headers: {header_dict}")
        # debug(f"EA API request 1st time: {time[0]} and value: {ssh[0]}")

        # Assign local dataset to object-scope dataset
        return dataset

    ############ BODC tide gauge methods ######################################
    def read_bodc(self, fn_bodc, date_start=None, date_end=None):
        """
        For reading from a single BODC (processed) file into an
        xarray dataset.
        If no data lies between the specified dates, a dataset is still created
        containing information on the tide gauge, but the time dimension will
        be empty.

        Data name: UK Tide Gauge Network, processed data.
        Source: https://www.bodc.ac.uk/
        See data notes from source for description of QC flags.

        The data takes the form:
            Port:              P234
            Site:              Liverpool, Gladstone Dock
            Latitude:          53.44969
            Longitude:         -3.01800
            Start Date:        01AUG2020-00.00.00
            End Date:          31AUG2020-23.45.00
            Contributor:       National Oceanography Centre, Liverpool
            Datum information: The data refer to Admiralty Chart Datum (ACD)
            Parameter code:    ASLVBG02 = Surface elevation (unspecified datum) 
            of the water body by bubbler tide gauge (second sensor)
              Cycle    Date      Time    ASLVBG02   Residual
             Number yyyy mm dd hh mi ssf         f          f
                 1) 2020/08/01 00:00:00     5.354M     0.265M
                 2) 2020/08/01 00:15:00     5.016M     0.243M
                 3) 2020/08/01 00:30:00     4.704M     0.241M
                 4) 2020/08/01 00:45:00     4.418M     0.255M
                 5) 2020/08/01 01:00:00     4.133      0.257
                 ...

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        """
        debug(f'Reading "{fn_bodc}" as a BODC file with {get_slug(self)}')  
        # TODO Maybe include start/end dates
        try:
            header_dict = self._read_bodc_header(fn_bodc)
            dataset = self._read_bodc_data(fn_bodc, date_start, date_end)
        except:
            raise Exception("Problem reading BODC file: " + fn_bodc)
        # Attributes
        dataset["longitude"] = header_dict["longitude"]
        dataset["latitude"] = header_dict["latitude"]
        del header_dict["longitude"]
        del header_dict["latitude"]

        dataset.attrs = header_dict
        self.dataset = dataset

    @staticmethod
    def _read_bodc_header(fn_bodc):
        """
        Reads header from a BODC file (format version 3.0).

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file

        Returns
        -------
        dictionary of attributes
        """
        debug(f'Reading BODC header from "{fn_bodc}"')
        fid = open(fn_bodc)

        # Read lines one by one (hopefully formatting is consistent)
        # Geographical stuff
        header_dict = {}
        header = True
        for line in fid:
            if ":" in line and header == True:
                (key, val) = line.split(":")
                key = key.lower().strip().replace(" ", "_")
                val = val.lower().strip().replace(" ", "_")
                header_dict[key] = val
                debug(f"Header key: {key} and value: {val}")
            else:
                # debug('No colon')
                header = False
        header_dict["site_name"] = header_dict["site"]  # duplicate as standard name
        debug(f'Read done, close file "{fn_bodc}"')
        fid.close()

        header_dict["latitude"] = float(header_dict["latitude"])
        header_dict["longitude"] = float(header_dict["longitude"])

        return header_dict

    @staticmethod
    def _read_bodc_data(fn_bodc, date_start=None, date_end=None, header_length: int = 11):
        """
        Reads observation data from a BODC file.

        Parameters
        ----------
        fn_bodc (str) : path to bodc tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, sealevel and quality control flags
        """
        # Initialise empty dataset and lists
        debug(f'Reading BODC data from "{fn_bodc}"')
        dataset = xr.Dataset()
        time = []
        ssh = []
        qc_flags = []
        residual = []
        # Open file and loop until EOF
        with open(fn_bodc) as file:
            line_count = 1
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count > header_length:
                    try:
                        working_line = line.split()
                        time_str = working_line[1] + " " + working_line[2]  # Empty lines cause trouble
                        ssh_str = working_line[3]
                        residual_str = working_line[4]
                        if ssh_str[-1].isalpha():
                            qc_flag_str = ssh_str[-1]
                            ssh_str = ssh_str.replace(qc_flag_str, "")
                            residual_str = residual_str.replace(qc_flag_str, "")
                        elif residual_str[-1].isalpha():  # sometimes residual has a
                            # flag when elevation does not
                            qc_flag_str = residual_str[-1]
                            ssh_str = ssh_str.replace(qc_flag_str, "")
                            residual_str = residual_str.replace(qc_flag_str, "")
                        else:
                            qc_flag_str = ""
                        # debug(line_count-header_length, residual_str, float(residual_str))
                        # debug(working_line, ssh_str, qc_flag_str)
                        time.append(time_str)
                        qc_flags.append(qc_flag_str)
                        ssh.append(float(ssh_str))
                        residual.append(float(residual_str))
                    except:
                        debug(f"{file} probably empty line at end. Breaks split()")
                line_count = line_count + 1
            debug(f'Read done, close file "{fn_bodc}"')

        # Convert time list to datetimes using pandas
        time = np.array(pd.to_datetime(time))

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            date_start = np.datetime64(date_start)
            start_index = np.argmax(time >= date_start)
        if date_end is not None:
            date_end = np.datetime64(date_end)
            end_index = np.argmax(time > date_end)
        time = time[start_index:end_index]
        ssh = ssh[start_index:end_index]
        qc_flags = qc_flags[start_index:end_index]

        # Set null values to nan
        ssh = np.array(ssh)
        qc_flags = np.array(qc_flags)
        # ssh[qc_flags==5] = np.nan

        # Assign arrays to Dataset
        dataset["ssh"] = xr.DataArray(ssh, dims=["time"]).expand_dims('id')
        dataset["qc_flags"] = xr.DataArray(qc_flags, dims=["time"]).expand_dims('id')
        dataset = dataset.assign_coords(time=("time", time))

        # Assign local dataset to object-scope dataset
        return dataset

    ##############################################################################
    ###                ~            Plotting             ~                     ###
    ##############################################################################

    def plot_timeseries(self, id, var_list=["ssh"], date_start=None, date_end=None, plot_line=False):
        """
        Quick plot of time series stored within object's dataset
        Parameters
        ----------
        date_start (datetime) : Start date for plotting
        date_end (datetime) : End date for plotting
        var_list  (str)  : List of variables to plot. Default: just ssh
        plot_line (bool) : If true, draw line between markers

        Returns
        -------
        matplotlib figure and axes objects
        """
        debug(f"Plotting timeseries for {get_slug(self)}")
        fig = plt.figure(figsize=(10, 10))
        # Check input is a list (even for one variable)
        if type(var_list) is str:
            var_list = [var_list]

        for var_str in var_list:
            dim_str = self.dataset[var_str].dims[0]
            x = np.array(self.dataset[dim_str])
            y = np.array(self.dataset[var_str])

            # Use only values between stated dates
            start_index = 0
            end_index = len(x)
            if date_start is not None:
                date_start = np.datetime64(date_start)
                start_index = np.argmax(x >= date_start)
            if date_end is not None:
                date_end = np.datetime64(date_end)
                end_index = np.argmax(x > date_end)
            x = x[start_index:end_index]
            y = y[start_index:end_index]

            # Plot lines first if needed
            if plot_line:
                plt.plot(x, y, c=[0.5, 0.5, 0.5], linestyle="--", linewidth=0.5)

            ax = plt.scatter(x, y, s=10)

        plt.grid()
        plt.xticks(rotation=45)
        plt.legend(var_list)
        # Title and axes
        plt.xlabel("Date")
        plt.title("Site: " + self.dataset.site_name)

        return fig, ax
    
    def plot_on_map(self):
        """
        Show the location of a tidegauge on a map.
        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()
        """
    
        debug(f"Plotting tide gauge locations for {get_slug(self)}")
    
        title = "Location: " + self.dataset.attrs["site_name"]
        X = self.dataset.longitude
        Y = self.dataset.latitude
        fig, ax = plot_util.geo_scatter(X, Y, title=title)
        ax.set_xlim((X - 10, X + 10))
        ax.set_ylim((Y - 10, Y + 10))
        return fig, ax
    
    @classmethod
    def plot_on_map_multiple(cls, tidegauge_list, color_var_str=None):
        """
        Show the location of a tidegauge on a map.
        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()
        """
    
        debug(f"Plotting tide gauge locations for {get_slug(cls)}")
    
        X = []
        Y = []
        C = []
        for tg in tidegauge_list:
            X.append(tg.dataset.longitude)
            Y.append(tg.dataset.latitude)
            if color_var_str is not None:
                C.append(tg.dataset[color_var_str].values)
    
        title = ""
        if color_var_str is None:
            fig, ax = plot_util.geo_scatter(X, Y, title=title)
        else:
            fig, ax = plot_util.geo_scatter(X, Y, title=title, c=C)
    
        ax.set_xlim((min(X) - 10, max(X) + 10))
        ax.set_ylim((min(Y) - 10, max(Y) + 10))
        return fig, ax

    ##############################################################################
    ###                ~        Model Comparison         ~                     ###
    ##############################################################################

    def crps(
        self,
        model_object,
        model_var_name,
        obs_var_name: str = "ssh",
        nh_radius: float = 20,
        time_interp: str = "linear",
        create_new_obj=True,
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

        mod_var = model_object.dataset[model_var_name]
        obs_var = self.dataset[obs_var_name]

        crps_list, n_model_pts, contains_land = crps_util.crps_sonf_fixed(
            mod_var,
            self.dataset.longitude.values,
            self.dataset.latitude.values,
            obs_var.values,
            obs_var.time.values,
            nh_radius,
            time_interp,
        )
        if create_new_obj:
            new_object = Tidegauge()
            new_dataset = self.dataset[["longitude", "latitude", "time"]]
            new_dataset["crps"] = (("time"), crps_list)
            new_dataset["crps_n_model_pts"] = (("time"), n_model_pts)
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset["crps"] = (("time"), crps_list)
            self.dataset["crps_n_model_pts"] = (("time"), n_model_pts)

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

        return_differenced = Tidegauge()
        return_differenced.dataset = differenced

        return return_differenced

    def absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        """Absolute difference two variables defined by var_str0 and var_str1
        between two dates date0 and date1. Return xr.DataArray"""
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.data_array_time_slice(var0, date0, date1).values
        var1 = general_utils.data_array_time_slice(var1, date0, date1).values
        adiff = np.abs(var0 - var1)
        return xr.DataArray(adiff, dims="time", name="absolute_error", coords={"time": self.dataset.time})

    def mean_absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        """Mean absolute difference two variables defined by var_str0 and
        var_str1 between two dates date0 and date1. Return xr.DataArray"""
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.data_array_time_slice(var0, date0, date1).values
        var1 = general_utils.data_array_time_slice(var1, date0, date1).values
        mae = metrics.mean_absolute_error(var0, var1)
        return mae

    def root_mean_square_error(self, var_str0, var_str1, date0=None, date1=None):
        """Root mean square difference two variables defined by var_str0 and
        var_str1 between two dates date0 and date1. Return xr.DataArray"""
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.data_array_time_slice(var0, date0, date1).values
        var1 = general_utils.data_array_time_slice(var1, date0, date1).values
        rmse = metrics.mean_squared_error(var0, var1)
        return np.sqrt(rmse)

    def time_mean(self, var_str, date0=None, date1=None):
        """Time mean of variable var_str between dates date0, date1"""
        var = self.dataset[var_str]
        var = general_utils.data_array_time_slice(var, date0, date1)
        return np.nanmean(var)

    def time_std(self, var_str, date0=None, date1=None):
        """Time st. dev of variable var_str between dates date0 and date1"""
        var = self.dataset[var_str]
        var = general_utils.data_array_time_slice(var, date0, date1)
        return np.nanstd(var)

    def time_correlation(self, var_str0, var_str1, date0=None, date1=None, method="pearson"):
        """Time correlation between two variables defined by var_str0,
        var_str1 between dates date0 and date1. Uses Pandas corr()."""
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename("var1")
        var1 = var1.rename("var2")
        var0 = general_utils.data_array_time_slice(var0, date0, date1)
        var1 = general_utils.data_array_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        corr = pdvar.corr(method=method)
        return corr.iloc[0, 1]

    def time_covariance(self, var_str0, var_str1, date0=None, date1=None):
        """Time covariance between two variables defined by var_str0,
        var_str1 between dates date0 and date1. Uses Pandas corr()."""
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename("var1")
        var1 = var1.rename("var2")
        var0 = general_utils.data_array_time_slice(var0, date0, date1)
        var1 = general_utils.data_array_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        cov = pdvar.cov()
        return cov.iloc[0, 1]

    def basic_stats(self, var_str0, var_str1, date0=None, date1=None, create_new_object=True):
        """Calculates a selection of statistics for two variables defined by
        var_str0 and var_str1, between dates date0 and date1. This will return
        their difference, absolute difference, mean absolute error, root mean
        square error, correlation and covariance. If create_new_object is True
        then this method returns a new TIDEGAUGE object containing statistics,
        otherwise variables are saved to the dateset inside this object."""

        diff = self.difference(var_str0, var_str1, date0, date1)
        ae = self.absolute_error(var_str0, var_str1, date0, date1)
        mae = self.mean_absolute_error(var_str0, var_str1, date0, date1)
        rmse = self.root_mean_square_error(var_str0, var_str1, date0, date1)
        corr = self.time_correlation(var_str0, var_str1, date0, date1)
        cov = self.time_covariance(var_str0, var_str1, date0, date1)

        if create_new_object:
            new_object = Tidegauge()
            new_dataset = self.dataset[["longitude", "latitude", "time"]]
            new_dataset["absolute_error"] = ae
            new_dataset["error"] = diff
            new_dataset["mae"] = mae
            new_dataset["rmse"] = rmse
            new_dataset["corr"] = corr
            new_dataset["cov"] = cov
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset["absolute_error"] = ae
            self.dataset["error"] = diff
            self.dataset["mae"] = mae
            self.dataset["rmse"] = rmse
            self.dataset["corr"] = corr
            self.dataset["cov"] = cov

    ##############################################################################
    ###                ~            Analysis             ~                     ###
    ##############################################################################

    def resample_mean(self, var_str: str, time_freq: str, **kwargs):
        """Resample a TIDEGAUGE variable in time by calculating the mean
            of all data points at a given frequency.

        Parameters
        ----------
        var_str (str)    : Variable name to resample
        time_freq (str)  : Time frequency. e.g. '1H' for hourly, '1D' for daily
                           Can also be a timedelta object. See Pandas resample
                           method for more info.
        **kwargs (other) : Other arguments to pass to xarray.Dataset.resample
        (http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html)

        Returns
        -------
        New variable (var_str_freq) and dimension (time_freq) in tg.dataset
        """
        # Define new variable and dimension names
        var = self.dataset[var_str]
        new_var_str = var_str + "_" + time_freq
        new_dim_str = "time_" + time_freq

        # Resample using xarray.resample
        resampled = var.resample(time=time_freq, **kwargs).mean()

        # Rename dimensions and variables. Put into original dataset.
        resampled = resampled.rename({"time": new_dim_str})
        resampled = resampled.rename(new_var_str)
        self.dataset[new_var_str] = resampled

    def apply_doodson_x0_filter(self, var_str):
        """Applies doodson X0 filter to a specified TIDEGAUGE variable
        Input ius expected to be hourly. Use resample_mean to average data
        to hourly frequency."""
        filtered = stats_util.doodson_x0_filter(self.dataset[var_str], ax=0)
        self.dataset[var_str + "_dx0"] = (("time_1H"), filtered)

    def find_high_and_low_water(self, var_str, method="comp", **kwargs):
        """
        Finds high and low water for a given variable.
        Returns in a new TIDEGAUGE object with similar data format to
        a TIDETABLE. If this Tidegauge object contains more than one location
        (id > 1) then a list of Tidegauges will be returned.

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

        dataset = self.dataset

        if 'id' in dataset.dims:
            n_id = dataset.dims['id']
        else:
            n_id = 1
            dataset = dataset.expand_dims('id')

        tg_list = []
        # Loop over id dimension
        for ii in range(n_id):
            # Get x and y for input to find_maxima
            
            x = dataset.isel(id=ii).time.values
            y = dataset.isel(id=ii)[var_str].values
    
            # Get time and values of maxima and minima
            time_max, values_max = stats_util.find_maxima(x, y, method=method, 
                                                          **kwargs)
            time_min, values_min = stats_util.find_maxima(x, -y, method=method, 
                                                          **kwargs)
            # Place the above values into a brand new dataset for this id index
            new_dataset = xr.Dataset()
            new_dataset.attrs = dataset.attrs
            new_dataset[var_str + "_highs"] = ("time_highs", values_max)
            new_dataset[var_str + "_lows"] = ("time_lows", -values_min)
            new_dataset["time_highs"] = ("time_highs", time_max)
            new_dataset["time_lows"] = ("time_lows", time_min)
    
            # Place dataset into a new Tidegauge object and append to output
            new_object = Tidegauge()
            new_object.dataset = new_dataset
            tg_list.append(new_object)
            
        # If only 1 index, return just a Tidegauge object, else return list.
        if n_id == 1:
            return new_object
        else:
            return tg_list

    def demean_timeseries(self):
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
            gridded.longitude, gridded.latitude, ds.longitude, 
            ds.latitude, mask=gridded.landmask
        )

        # Extract spatial time series
        print("Calculating time indices.", flush=True)
        extracted = gridded.isel(x_dim=ind_x, y_dim=ind_y)
        if 'dim_0' in extracted.dims:
            extracted = extracted.swap_dims({"dim_0": "id"})
        else:
            extracted = extracted.expand_dims('id')
            
        print(extracted)

        # Compute data (takes a while..)
        print(" Indexing model data at tide gauge locations.. ", flush=True)
        extracted.load()

        # Check interpolation distances
        print("Calculating interpolation distances.", flush=True)
        interp_dist = general_utils.calculate_haversine_distance(
            extracted.longitude, extracted.latitude, 
            ds.longitude.values, ds.latitude.values
        )

        # Interpolate model onto obs times
        print("Interpolating in time...", flush=True)
        extracted = extracted.rename({"time": "t_dim"})
        extracted = extracted.interp(t_dim=ds.time.values, method=time_interp)

        # Put interp_dist into dataset
        extracted["interp_dist"] = interp_dist
        extracted = extracted.rename_vars({"t_dim": "time"})

        tg_out = Tidegauge()
        tg_out.dataset = extracted
        return tg_out
