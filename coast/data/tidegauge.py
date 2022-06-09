"""Tide Gauge class"""
from .timeseries import Timeseries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import pytz
from .._utils import general_utils, plot_util, crps_util, stats_util
from .._utils.logging_util import get_slug, debug, error, info
from typing import Union
from pathlib import Path


class Tidegauge(Timeseries):
    """
    This is an object for storage and manipulation of tide gauge data
    in a single dataset. This may require some processing of the observations
    such as interpolation to a common time step.

    This object's dataset should take the form (as with Timeseries):

        Dimensions:
            id_dim   : The locations dimension. Each time series has an index
            time : The time dimension. Each datapoint at each port has an index

        Coordinates:
            longitude (id_dim) : Longitude values for each port index
            latitude  (id_dim) : Latitude values for each port index
            time      (time) : Time values for each time index (datetime)
            id_name   (id_dim)   : Name of index, e.g. port name or mooring id.

    An example data variable could be ssh, or ntr (non-tidal residual). This
    object can also be used for other instrument types, not just tide gauges.
    For example moorings.

    Every id index for this object should use the same time coordinates.
    Therefore, timeseries need to be aligned before being placed into the
    object. If there is any padding needed, then NaNs should be used. NaNs
    should also be used for quality control/data rejection.
    """

    def __init__(self, dataset=None, config: Union[Path, str] = None, new_time_coords=None):
        """
        Initialise TIDEGAUGE object as empty or by providing an existing
        dataset or tidegauge object. There are read functions within Tidegauge()
        which can subsequently read into this object's dataset.'

        Example usage:
        --------------
        # Create empty tidegauge object
        tidegauge = coast.Tidegauge()

        # Create new tidegauge object containing an existing xarray dataset
        tidegauge = coast.Tidegauge(dataset = dataset_name)

        # Create new tidegauge object containing existing xarray dataset and
        # create new time coordinate (will remove all variables)
        tidegauge = coast.Tidegage(tidegauge0.dataset, new_coords = time_array)

        INPUTS
        ----------
        dataset (xr.Dataset)    :: Xarray dataset to insert into new object
        config  (Path or Str)   :: Configuration file to use when calling
                                   read functions (saved in object) [Optional]
        new_time_coords (array) :: New time coords to create [Optional]

        Returns
        -------
        New Tidegauge object.
        """
        debug(f"Creating a new ..... {get_slug(self)}")
        super().__init__(config)

        # If file list is supplied, read files from directory
        if dataset is not None:
            self.dataset = dataset.copy()
            self.apply_config_mappings()

            # If new_time_coords, replace existing time dimension with new
            if new_time_coords is not None:
                ds_coords = xr.Dataset(self.dataset.coords).drop("time")
                ds_coords["time"] = ("t_dim", new_time_coords)
                ds_coords = ds_coords.set_coords("time")
                self.dataset = ds_coords.copy()

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
        dataset = None

        # See if its a file list input, or a glob
        if type(fn_gesla) is not list:
            file_list = glob.glob(fn_gesla)
        else:
            file_list = fn_gesla

        multiple = False
        if len(file_list) > 1:
            multiple = True

        ds_list = []
        # Loop over files and put resulting dataset into an output list
        for fn in file_list:
            try:
                header_dict = self._read_gesla_header_v3(fn)
                dataset = self._read_gesla_data_v3(fn, date_start, date_end)
            except:
                raise Exception("Problem reading GESLA file: " + fn)
            # Attributes
            dataset["longitude"] = ("id_dim", [header_dict["longitude"]])
            dataset["latitude"] = ("id_dim", [header_dict["latitude"]])
            dataset["id_name"] = ("id_dim", [header_dict["site_name"]])
            dataset = dataset.set_coords(["longitude", "latitude", "id_name"])

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
    def _read_gesla_data_v3(cls, fn_gesla, date_start=None, date_end=None, header_length: int = 32):
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
        dataset["ssh"] = xr.DataArray(ssh, dims=["t_dim"]).expand_dims("id_dim")
        dataset["qc_flags"] = xr.DataArray(qc_flags, dims=["t_dim"]).expand_dims("id_dim")
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
        header_dict = {"site_name": site_name, "field": field, "units": units, "datum": datum}
        return header_dict

    @classmethod
    def _read_hlw_data(cls, filnam, header_dict, date_start=None, date_end=None, header_length: int = 1):
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
                            bst_obj = pytz.timezone("Europe/London")
                            time.append(np.datetime64(bst_obj.localize(datetime_obj).astimezone(pytz.utc)))
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
        dataset["ssh"] = xr.DataArray(ssh, dims=["time"]).expand_dims("id_dim")
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
                    np.datetime_as_string(self.dataset.time[i], unit="m", timezone=pytz.timezone(timezone)),
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

            ssh = self.dataset[measure_var].isel(time=slice(start_index, end_index))

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
        dataset["ssh"] = xr.DataArray(ssh, dims=["time"]).expand_dims("id_dim")
        dataset["qc_flags"] = xr.DataArray(qc_flags, dims=["time"]).expand_dims("id_dim")
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

        X = self.dataset.longitude
        Y = self.dataset.latitude
        fig, ax = plot_util.geo_scatter(X, Y)
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

        # Determine if landmask is present
        if "landmask" not in gridded:
            gridded.landmask = None

        # Determine spatial indices
        print("Calculating spatial indices.", flush=True)
        ind_x, ind_y = general_utils.nearest_indices_2d(
            gridded.longitude, gridded.latitude, ds.longitude, ds.latitude, mask=gridded.landmask
        )

        # Extract spatial time series
        print("Calculating time indices.", flush=True)
        extracted = gridded.isel(x_dim=ind_x, y_dim=ind_y)
        if "dim_0" in extracted.dims:
            extracted = extracted.swap_dims({"dim_0": "id_dim"})
        else:
            extracted = extracted.expand_dims("id_dim")

        # Compute data (takes a while..)
        print(" Indexing model data at tide gauge locations.. ", flush=True)
        extracted.load()

        # Check interpolation distances
        print("Calculating interpolation distances.", flush=True)
        interp_dist = general_utils.calculate_haversine_distance(
            extracted.longitude.values, extracted.latitude.values, ds.longitude.values, ds.latitude.values
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

    def time_slice(self, date0, date1):
        """Return new Gridded object, indexed between dates date0 and date1"""
        dataset = self.dataset
        t_ind = pd.to_datetime(dataset.time.values) >= date0
        dataset = dataset.isel(t_dim=t_ind)
        t_ind = pd.to_datetime(dataset.time.values) < date1
        dataset = dataset.isel(t_dim=t_ind)
        return Tidegauge(dataset=dataset)

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Get a subset of this Profile() object in a spatial box.

        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]

        return: A new profile object containing subsetted data
        """
        ind = general_utils.subset_indices_lonlat_box(
            self.dataset.longitude, self.dataset.latitude, lonbounds[0], lonbounds[1], latbounds[0], latbounds[1]
        )
        return Tidegauge(dataset=self.dataset.isel(id_dim=ind[0]))
