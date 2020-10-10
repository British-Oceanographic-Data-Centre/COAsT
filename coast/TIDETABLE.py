import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sklearn.metrics as metrics
from . import general_utils, plot_util, crps_util
from .logging_util import get_slug, debug, error, info
import datetime
import re

def npdatetime64_2_datetime(date):
    """
    Convert from numpy.dateime64 to datetime
    Helpful guidance: https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    """
    print('type(in_date):',type(date))
    dt64 = np.datetime64(date) # already this type, but just to be explicit
    print('type(dt64):',type(dt64))
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (dt64 - unix_epoch) / one_second
    print('type(out_date):',type(datetime.datetime.utcfromtimestamp(seconds_since_epoch)))
    return datetime.datetime.utcfromtimestamp(seconds_since_epoch)

def nearest_datetime_ind(items, pivot):
    """
    find the index from items for the nearest value to pivot
    This method should surely be stored somewhere else...

    items - an array of timezone aware datetime objects
       E.g. array([datetime.datetime(2020, 1, 1, 2, 36, tzinfo=datetime.timezone(datetime.timedelta(0), 'GMT')),
        datetime.datetime(2020, 1, 1, 21, 41, tzinfo=datetime.timezone(datetime.timedelta(0), 'GMT'))],
        dtype=object)
    """
    #debug("nearest_datetime_ind",pivot.tzinfo)
    #debug(type(pivot.tzinfo))
    #debug( [date.tzinfo for date in items] )
    #debug( [type(date.tzinfo) for date in items] )
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)

class TIDETABLE(object):
    """
    This is where the main things happen.
    Where user input is managed and methods are launched
    """
    def __init__(self, file_path = None, \
                date_start = None, \
                date_end = None):
        '''
        Initialise TIDEGAUGE object either as empty (no arguments) or by
        reading HLW data from a directory between two datetime objects.

        date objects can be datetime or np.datetime64

        # Read tide gauge data for 2020
        filnam = '/Users/jeff/GitHub/DeeBore/data/Liverpool_2015_2020_HLW.txt'
        date_start = datetime.datetime(2020,1,1)
        date_end = datetime.datetime(2020,12,31)
        tg = TIDETABLE(filnam, date_start, date_end)

        # Access the data
        tg.dataset

        # Exaple plot
        tg.dataset.plot.scatter(x="time", y="sea_level")

        '''
        # Ensure the date objects are np.datetime64 and days
        if date_start is not None:
            date_start = np.datetime64(date_start, 'D')
        if date_end is not None:
            date_end = np.datetime64(date_end, 'D') + np.timedelta64(1,'D')

        # If file list is supplied, read files from directory
        if file_path is None:
            self.dataset = None
        else:
            self.dataset = self.read_HLW_to_xarray(file_path, \
                                                date_start , \
                                                date_end )
        return

    @classmethod
    def read_HLW_to_xarray(cls, filnam, date_start:np.datetime64=None, \
                                        date_end:np.datetime64=None):
        '''

        Parameters
        ----------
        filnam (str) : path to gesla tide gauge file
        date_start (np.datetime64) : start date for returning data
        date_end (np.datetime64) : end date for returning data

        Returns
        -------
        xarray.Dataset object.
        '''
        try:
            header_dict = cls.read_HLW_header(filnam)
            dataset = cls.read_HLW_data(filnam, header_dict, date_start, date_end)
            if header_dict['field'] == 'TZ:UT(GMT)/BST':
                info('Read in as BST, stored as UTC')
            elif header_dict['field'] == 'TZ:GMTonly':
                info('Read and store as GMT/UTC')
            else:
                debug("Not expecting that timezone")

        except:
            raise Exception('Problem reading HLW file: ' + filnam)

        dataset.attrs = header_dict

        return dataset


    @staticmethod
    def read_HLW_header(filnam):
        '''
        Reads header from a HWL file.

        Parameters
        ----------
        filnam (str) : path to file

        Returns
        -------
        dictionary of attributes
        '''
        info(f"Reading HLW header from \"{filnam}\" ")
        fid = open(filnam)

        # Read lines one by one (hopefully formatting is consistent)
        header = re.split( r"\s{2,}", fid.readline() )
        site_name = header[0]
        site_name = site_name.replace(' ','')

        field = header[1]
        field = field.replace(' ','')

        units = header[2]
        units = units.replace(' ','')

        datum = header[3]
        datum = datum.replace(' ','')

        if(0):
            header = fid.readline().split()
            site_name = header[:3]
            site_name = '_'.join(site_name)

            field = header[3:5]
            field = '_'.join(field).replace(':_',':')

            units = header[5:7]
            units = '_'.join(units).replace(':_',':')

            datum = header[7:10]
            datum = '_'.join(datum).replace(':_',':')

        info(f"Read done, close file \"{filnam}\"")
        fid.close()
        # Put all header info into an attributes dictionary
        header_dict = {'site_name' : site_name, 'field':field,
                       'units':units, 'datum':datum}
        return header_dict

    @staticmethod
    def read_HLW_data(filnam, header_dict, date_start=None, date_end=None,
                           header_length:int=1):
        '''
        Reads observation data from a GESLA file (format version 3.0).

        Parameters
        ----------
        filnam (str) : path to HLW tide gauge file
        date_start (np.datetime64) : start date for returning data.
        date_end (np.datetime64) : end date for returning data.
        header_length (int) : number of lines in header (to skip when reading)

        Returns
        -------
        xarray.Dataset containing times, High and Low water values
        '''
        # Initialise empty dataset and lists
        info(f"Reading HLW data from \"{filnam}\"")
        dataset = xr.Dataset()
        time = []
        sea_level = []

        if header_dict['field'] == 'TZ:UT(GMT)/BST':
            localtime_flag = True
            #if date_start is not None: date_start = date_start.astimezone()
            #if date_end is not None: date_end = date_end.astimezone()
        else:
            localtime_flag = False

        # Open file and loop until EOF
        with open(filnam) as file:
            line_count = 0
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count>header_length:
                    working_line = line.split()
                    if working_line[0] != '#':
                        time_str = working_line[0] + ' ' + working_line[1]
                        # Read time as datetime.datetime because it can handle local timezone easily
                        datetime_obj = datetime.datetime.strptime( time_str , '%d/%m/%Y %H:%M')
                        if localtime_flag == True:
                            time.append( np.datetime64(datetime_obj.astimezone() ))
                        else:
                            time.append( np.datetime64(datetime_obj) )
                        sea_level.append(float(working_line[2]))

                line_count = line_count + 1
            info(f"Read done, close file \"{filnam}\"")

        # Return only values between stated dates
        start_index = 0
        end_index = len(time)


        if date_start is not None:
            start_index = nearest_datetime_ind(time, date_start)
        if date_end is not None:
            end_index = nearest_datetime_ind(time, date_end)
        time = time[start_index:end_index+1]
        sea_level = sea_level[start_index:end_index+1]

        # Assign arrays to Dataset
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['t_dim'])
        dataset = dataset.assign_coords(time = ('t_dim', time))

        #info('Time zone type', type(time[0].tzinfo) )

        # Assign local dataset to object-scope dataset
        return dataset

    def get_tidetabletimes(self, time_guess:datetime = None, window:int = 2):
        """
        Get tide times and heights from tide table.
        input:
        time_guess : np.datetime64 or datetime
                assumes utc
        window:  +/- hours window size (int)

        returns:
        height (m), time (utc)

        The function nearest_datetime_ind makes the window searching a bit irrelevant for short windows
        """

        # Ensure the date objects are datetime
        if type(time_guess) is not np.datetime64:
            info('Convert date to np.datetime64')
            time_guess = np.datetime64(time_guess)

        if time_guess == None:
            info("Use today's date")
            time_guess = np.datetime64('now')

        # Return only values between stated dates
        start_index = 0
        end_index = len(self.dataset.time)

        debug(f"test: {time_guess - np.timedelta64(window, 'h')}")
        #print('2h', type(datetime.timedelta(hours=2).tzinfo))

        start_index = nearest_datetime_ind(self.dataset.time.values, time_guess - np.timedelta64(window, 'h'))
        #if self.dataset.time[start_index] > time_guess: start_index = start_index - 1
        end_index  =  nearest_datetime_ind(self.dataset.time.values, time_guess + np.timedelta64(window, 'h'))
        #if self.dataset.time[end_index] < time_guess: end_index = end_index + 1

        debug(f"time_guess - win: {time_guess-np.timedelta64(window, 'h')}")
        debug(f"time[start_index-1:+1]: {self.dataset.time.values[start_index-1:start_index+1]}")

        time = self.dataset.time[start_index:end_index+1].values
        sea_level = self.dataset.sea_level[start_index:end_index+1].values

        return sea_level, time

##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################

    def plot_on_map(self):
        '''
        Show the location of a tidegauge on a map.

        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()

        '''

        debug(f"Plotting tide gauge locations for {get_slug(self)}")

        title = 'Location: ' + self.dataset.attrs['site_name']
        X = self.dataset.longitude
        Y = self.dataset.latitude
        fig, ax =  plot_util.geo_scatter(X, Y, title=title,
                                         xlim = [X-10, X+10],
                                         ylim = [Y-10, Y+10])
        return fig, ax

    @classmethod
    def plot_on_map_multiple(cls,tidegauge_list, color_var_str = None):
        '''
        Show the location of a tidegauge on a map.

        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()

        '''

        debug(f"Plotting tide gauge locations for {get_slug(cls)}")

        X = []
        Y = []
        C = []
        for tg in tidegauge_list:
            X.append(tg.dataset.longitude)
            Y.append(tg.dataset.latitude)
            if color_var_str is not None:
                C.append(tg.dataset[color_var_str])
        title = ''

        if color_var_str is None:
            fig, ax =  plot_util.geo_scatter(X, Y, title=title,
                                             xlim = [min(X)-10, max(X)+10],
                                             ylim = [min(Y)-10, max(Y)+10])
        else:
            fig, ax =  plot_util.geo_scatter(X, Y, title=title,
                                             colors = C,
                                             xlim = [X-10, X+10],
                                             ylim = [Y-10, Y+10])
        return fig, ax

    def plot_timeseries(self, var_name = 'sea_level',
                        date_start=None, date_end=None,
                        qc_colors=True,
                        plot_line = False):
        '''
        Quick plot of time series stored within object's dataset
        Parameters
        ----------
        date_start (datetime) : Start date for plotting
        date_end (datetime) : End date for plotting
        var_name (str) : Variable to plot. Default: sea_level
        qc_colors (bool) : If true, markers are coloured according to qc values
        plot_line (bool) : If true, draw line between markers

        Returns
        -------
        matplotlib figure and axes objects
        '''
        debug(f"Plotting timeseries for {get_slug(self)}")
        x = np.array(self.dataset.time)
        y = np.array(self.dataset[var_name])
        qc = np.array(self.dataset.qc_flags)
        # Use only values between stated dates
        start_index = 0
        end_index = len(x)
        if date_start is not None:
            date_start = np.datetime64(date_start)
            start_index = np.argmax(x>=date_start)
        if date_end is not None:
            date_end = np.datetime64(date_end)
            end_index = np.argmax(x>date_end)
        x = x[start_index:end_index]
        y = y[start_index:end_index]
        qc = qc[start_index:end_index]

        # Plot lines first if needed
        if plot_line:
            plt.plot(x,y, c=[0.5,0.5,0.5], linestyle='--', linewidth=0.5)

        # Two plotting routines for whether or not to use qc flags.
        if qc_colors:
            size = 5
            fig = plt.figure(figsize=(10,10))
            ax = plt.scatter(x[qc==0], y[qc==0], s=size)
            plt.scatter(x[qc==1], y[qc==1], s=size)
            plt.scatter(x[qc==2], y[qc==2], s=size)
            plt.scatter(x[qc==3], y[qc==3], s=size)
            plt.scatter(x[qc==4], y[qc==4], s=size)
            plt.grid()
            plt.legend(['No QC','Correct','Interpolated','Doubtful','Spike'],
                       loc='upper left', ncol=5)
            plt.xticks(rotation=45)
        else:
            fig = plt.figure(figsize=(10,10))
            plt.scatter(x,y)
            plt.grid()
            plt.xticks(rotation=65)

        # Title and axes
        plt.xlabel('Date')
        plt.ylabel(var_name + ' (m)')
        plt.title(var_name + ' at site: ' + self.dataset.site_name)

        return fig, ax

##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################

    def obs_operator(self, model, mod_var_name:str, time_interp = 'nearest'):
        '''
        Interpolates a model array (specified using a model object and variable
        string) to TIDEGAUGE location and times. Takes the nearest model grid
        cell to the tide gauge.

        Parameters
        ----------
        model : MODEL object (e.g. NEMO)
        model_var_name (str) : Name of variable (inside MODEL) to interpolate.
        time_interp (str) : type of scipy time interpolation (e.g. linear)

        Returns
        -------
        Saves interpolated array to TIDEGAUGE.dataset
        '''

        # Get data arrays
        mod_var_array = model.dataset[mod_var_name]

        # Depth interpolation -> for now just take 0 index
        if 'z_dim' in mod_var_array.dims:
            mod_var_array = mod_var_array.isel(z_dim=0).squeeze()

        # Cast lat/lon to numpy arrays
        obs_lon = np.array([self.dataset.longitude])
        obs_lat = np.array([self.dataset.latitude])

        interpolated = model.interpolate_in_space(mod_var_array, obs_lon,
                                                  obs_lat)

        interpolated = model.interpolate_in_time(interpolated,
                                                 self.dataset.time)

        # Store interpolated array in dataset
        new_var_name = 'interp_' + mod_var_name
        self.dataset[new_var_name] = interpolated.drop(['longitude','latitude'])
        return

    def crps(self, model_object, model_var_name, obs_var_name:str='sea_level',
         nh_radius: float = 20, cdf_type:str='empirical',
         time_interp:str='linear', create_new_obj = True):
        '''
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
        # Compare modelled 'sossheig' with 'sea_level' using CRPS
        crps = altimetry.crps(nemo, 'sossheig', 'sea_level')
        '''

        mod_var = model_object.dataset[model_var_name]
        obs_var = self.dataset[obs_var_name]

        crps_list, n_model_pts, contains_land = crps_util.crps_sonf_fixed(
                               mod_var,
                               self.dataset.longitude,
                               self.dataset.latitude,
                               obs_var.values,
                               obs_var.time.values,
                               nh_radius, cdf_type, time_interp )
        if create_new_obj:
            new_object = TIDETABLE()
            new_dataset = self.dataset[['longitude','latitude','time']]
            new_dataset['crps'] =  (('t_dim'),crps_list)
            new_dataset['crps_n_model_pts'] = (('t_dim'), n_model_pts)
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset['crps'] =  (('t_dim'),crps_list)
            self.dataset['crps_n_model_pts'] = (('t_dim'), n_model_pts)

    def difference(self, var_str0:str, var_str1:str, date0=None, date1=None):
        ''' Difference two variables defined by var_str0 and var_str1 between
        two dates date0 and date1. Returns xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        diff = var0 - var1
        return xr.DataArray(diff, dims='t_dim', name='error',
                            coords={'time':self.dataset.time})

    def absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Absolute difference two variables defined by var_str0 and var_str1
        between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        adiff = np.abs(var0 - var1)
        return xr.DataArray(adiff, dims='t_dim', name='absolute_error',
                            coords={'time':self.dataset.time})

    def mean_absolute_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Mean absolute difference two variables defined by var_str0 and
        var_str1 between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        mae = metrics.mean_absolute_error(var0, var1)
        return mae

    def root_mean_square_error(self, var_str0, var_str1, date0=None, date1=None):
        ''' Root mean square difference two variables defined by var_str0 and
        var_str1 between two dates date0 and date1. Return xr.DataArray '''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = general_utils.dataarray_time_slice(var0, date0, date1).values
        var1 = general_utils.dataarray_time_slice(var1, date0, date1).values
        rmse = metrics.mean_squared_error(var0, var1)
        return np.sqrt(rmse)

    def time_mean(self, var_str, date0=None, date1=None):
        ''' Time mean of variable var_str between dates date0, date1'''
        var = self.dataset[var_str]
        var = general_utils.dataarray_time_slice(var, date0, date1)
        return np.nanmean(var)

    def time_std(self, var_str, date0=None, date1=None):
        ''' Time st. dev of variable var_str between dates date0 and date1'''
        var = self.dataset[var_str]
        var = general_utils.dataarray_time_slice(var, date0, date1)
        return np.nanstd(var)

    def time_correlation(self, var_str0, var_str1, date0=None, date1=None,
                         method='pearson'):
        ''' Time correlation between two variables defined by var_str0,
        var_str1 between dates date0 and date1. Uses Pandas corr().'''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename('var1')
        var1 = var1.rename('var2')
        var0 = general_utils.dataarray_time_slice(var0, date0, date1)
        var1 = general_utils.dataarray_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        corr = pdvar.corr(method=method)
        return corr.iloc[0,1]

    def time_covariance(self, var_str0, var_str1, date0=None, date1=None):
        ''' Time covariance between two variables defined by var_str0,
        var_str1 between dates date0 and date1. Uses Pandas corr().'''
        var0 = self.dataset[var_str0]
        var1 = self.dataset[var_str1]
        var0 = var0.rename('var1')
        var1 = var1.rename('var2')
        var0 = general_utils.dataarray_time_slice(var0, date0, date1)
        var1 = general_utils.dataarray_time_slice(var1, date0, date1)
        pdvar = xr.merge((var0, var1))
        pdvar = pdvar.to_dataframe()
        cov = pdvar.cov()
        return cov.iloc[0,1]

    def basic_stats(self, var_str0, var_str1, date0 = None, date1 = None,
                    create_new_object = True):
        ''' Calculates a selection of statistics for two variables defined by
        var_str0 and var_str1, between dates date0 and date1. This will return
        their difference, absolute difference, mean absolute error, root mean
        square error, correlation and covariance. If create_new_object is True
        then this method returns a new TIDEGAUGE object containing statistics,
        otherwise variables are saved to the dateset inside this object. '''

        diff = self.difference(var_str0, var_str1, date0, date1)
        ae = self.absolute_error(var_str0, var_str1, date0, date1)
        mae = self.mean_absolute_error(var_str0, var_str1, date0, date1)
        rmse = self.root_mean_square_error(var_str0, var_str1, date0, date1)
        corr = self.time_correlation(var_str0, var_str1, date0, date1)
        cov = self.time_covariance(var_str0, var_str1, date0, date1)

        if create_new_object:
            new_object = TIDETABLE()
            new_dataset = self.dataset[['longitude','latitude','time']]
            new_dataset['absolute_error'] = ae
            new_dataset['error'] = diff
            new_dataset['mae'] = mae
            new_dataset['rmse'] = rmse
            new_dataset['corr'] = corr
            new_dataset['cov'] = cov
            new_object.dataset = new_dataset
            return new_object
        else:
            self.dataset['absolute_error'] = ae
            self.dataset['error'] = diff
            self.dataset['mae'] = mae
            self.dataset['rmse'] = rmse
            self.dataset['corr'] = corr
            self.dataset['cov'] = cov
