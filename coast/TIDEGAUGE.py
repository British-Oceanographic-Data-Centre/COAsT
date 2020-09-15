import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
from warnings import warn

class TIDEGAUGE():
    '''
    An object for reading, storing and manipulating tide gauge data.
    Reading and organisation methods are centred around the GESLA database.
    However, any fixed time series data can be used if in the correct format.
    (Source: https://www.gesla.org/)
    The data format used for this object is as follows:
        
    *Data Format Overview*
        
        1. Data for a single tide gauge is stored in an xarray Dataset object.
           This can be accessed using TIDEGAUGE.dataset (replacing TIDEGAUGE
           with an objects local name). 
        2. The dataset has a single dimension: time.
        3. Latitude/Longitude and other single values parameters are stored as
           attributes.
        4. Sea_level, quality control flags, time and similar are stored as
           variables along the time dimension.
           
    *Methods Overview*
    
        1. __init__: Can be initialised with a GESLA file or empty.
        2. plot_map: Plots locations of all time series on a map.
        3. plot_timeseries: Plots a specified time series.
        4. obs_operator: Interpolates model data to time series locations
           and times (not yet implemented).
        5. read_gesla_to_xarray_v3: Reads a format version 3.0 
           GESLA file to an xarray Dataset.
        6. read_gesla_header_v3: Reads the header of a version 3 
           GESLA file.
        7. read_gesla_data_v3: Reads data from a version 3 GESLA 
           file.
        8. create_tidegauge_multiple: Creates multiple tide gauge 
           objects from a list of filenames or directory and returns them 
           in a list.
           
    There are some methods outside of the class but still relevant to the
    TIDEGAUGE object. Please see the *Routines* section of this file.
    '''  
    
    def __init__(self, file_path = None, date_start=None, date_end=None):
        '''
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

        Returns
        -------
        Self
        '''
        
        # If file list is supplied, read files from directory
        if file_path is None:
            self.dataset = None
        else:
            self.dataset = self.read_gesla_to_xarray_v3(file_path, 
                                                        date_start, date_end)
        return
    
    def plot_on_map(self):
        '''
        Show the location of a tidegauge on a map.
        
        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()

        '''
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        except ImportError:
            import sys
            warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
            sys.exit(-1)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        X = self.dataset.longitude
        Y = self.dataset.latitude

        cset = plt.scatter(X, Y, c='r')

        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        coast = NaturalEarthFeature(category='physical', scale='50m',
                                    facecolor=[0.8,0.8,0.8], name='coastline',
                                    alpha=0.5)
        ax.add_feature(coast, edgecolor='gray')
        plt.title('Tide gauge at site: ' + self.dataset.site_name)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.xlim(X-10, X+10)
        plt.ylim(Y-10, Y+10)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='-')

        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.show()
        
        return fig, ax
    
    def plot_timeseries(self, date_start=None, date_end=None, 
                        var_name = 'sea_level', qc_colors=True, 
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
        
        interpolated = model.interpolate_in_space(mod_var_array, obs_lon, obs_lat)
        
        interpolated = model.interpolate_in_time(interpolated, self.dataset.time)

        # Store interpolated array in dataset
        new_var_name = 'interp_' + mod_var_name
        self.dataset[new_var_name] = interpolated
        return
    
    @classmethod
    def read_gesla_to_xarray_v3(cls, fn_gesla, date_start=None, date_end=None):
        '''
        For reading from a single GESLA2 (Format version 3.0) file into an
        xarray dataset. Formatting according to Woodworth et al. (2017).
        Website: https://www.gesla.org/
        If no data lies between the specified dates, a dataset is still created
        containing information on the tide gauge, but the time dimension will
        be empty.
        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
          
        Returns
        -------
        xarray.Dataset object.
        '''
        try:
            header_dict = cls.read_gesla_header_v3(fn_gesla)
            dataset = cls.read_gesla_data_v3(fn_gesla, date_start, date_end)
        except:
            raise Exception('Problem reading GESLA file: ' + fn_gesla)
        # Attributes
        dataset.attrs = header_dict
        
        return dataset
    
    @classmethod
    def create_multiple_tidegauge(cls, file_list, date_start=None, 
                                  date_end=None):
        '''
        Initialise TIDEGAUGE object either as empty (no arguments) or by
        reading GESLA data from a directory between two datetime objects.
    
        Example usage:
            --------------
            # Read all data in directory in January 1990
            date0 = datetime.datetime(1990,1,1)
            date1 = datetime.datetime(1990,2,1)
            tg = coast.TIDEGAUGE('gesla_directory/', date0, date1)
            
            Parameters
            ----------
            directory (str) : Path to directory containing desired GESLA files
            file_list (list of str) : list of filenames to read from directory.
            Optional.
            date_start (datetime) : Start date for data read. Optional
            date_end (datetime) : end date for data read. Optional

        Returns
        -------
        List of TIDEGAUGE objects.
        '''
        # If single string is given then put into a single element list
        if type(file_list) is str:
            file_list = [file_list]
            
        # Check file_list for wildcards and make list of files to read
        file_to_read = []
        for file in file_list:
            if '*' in file:
                wildcard_list = glob.glob(file)
                file_to_read = file_to_read + wildcard_list
            else:
                file_to_read.append(file)
            
        # Loop over files to read and read them into datasets
        tidegauge_list = []
        #longitude_list = []
        #latitude_list = []
        #sitename_list = []
        for file in file_to_read:
            try:
                dataset = cls.read_gesla_to_xarray_v3(file, date_start, 
                                                      date_end)
                tidegauge_list.append(dataset)
            except:
                # Problem with reading file: file
                pass
        return tidegauge_list
    
    @staticmethod
    def read_gesla_header_v3(fn_gesla):
        '''
        Reads header from a GESLA file (format version 3.0).
            
        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file
          
        Returns
        -------
        dictionary of attributes
        '''
        fid = open(fn_gesla)
        
        # Read lines one by one (hopefully formatting is consistent)
        fid.readline() # Skip first line
        # Geographical stuff
        site_name = fid.readline().split()[3:]
        site_name = '_'.join(site_name)
        country = fid.readline().split()[2:]
        country = '_'.join(country)
        contributor = fid.readline().split()[2:]
        contributor = '_'.join(contributor)
        # Coordinates
        latitude = float(fid.readline().split()[2])
        longitude = float(fid.readline().split()[2])
        coordinate_system = fid.readline().split()[3]
        # Dates
        start_date = fid.readline().split()[3:5] 
        start_date = ' '.join(start_date)
        start_date = pd.to_datetime(start_date)
        end_date = fid.readline().split()[3:5]
        end_date = ' '.join(end_date)
        end_date = pd.to_datetime(end_date)
        time_zone_hours = float(fid.readline().split()[4])
        # Other
        fid.readline() #Datum
        fid.readline() #Instrument
        precision = float(fid.readline().split()[2])
        null_value = float( fid.readline().split()[3])
        
        fid.close()
        # Put all header info into an attributes dictionary
        header_dict = {'site_name' : site_name, 'country':country, 
                       'contributor':contributor, 'latitude':latitude,
                       'longitude':longitude, 
                       'coordinate_system':coordinate_system,
                       'original_start_date':start_date, 
                       'original_end_date': end_date,
                       'time_zone_hours':time_zone_hours, 
                       'precision':precision, 'null_value':null_value}
        return header_dict
        
    @staticmethod
    def read_gesla_data_v3(cls, fn_gesla, date_start=None, date_end=None,
                           header_length:int=32):
        '''
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
        '''
        # Initialise empty dataset and lists
        dataset = xr.Dataset()
        time = []
        sea_level = []
        qc_flags = []
        # Open file and loop until EOF
        with open(fn_gesla) as file:
            line_count = 0
            for line in file:
                # Read all data. Date boundaries are set later.
                if line_count>header_length:
                    working_line = line.split()
                    if working_line[0] != '#':
                        time.append(working_line[0] + ' ' + working_line[1])
                        sea_level.append(float(working_line[2]))
                        qc_flags.append(int(working_line[3]))
                    
                line_count = line_count + 1
                
        # Convert time list to datetimes using pandas
        time = np.array(pd.to_datetime(time))
        
        # Return only values between stated dates
        start_index = 0
        end_index = len(time)
        if date_start is not None:
            date_start = np.datetime64(date_start)
            start_index = np.argmax(time>=date_start)
        if date_end is not None:
            date_end = np.datetime64(date_end)
            end_index = np.argmax(time>date_end)
        time = time[start_index:end_index]
        sea_level = sea_level[start_index:end_index]
        qc_flags=qc_flags[start_index:end_index]
        
        # Set null values to nan
        sea_level = np.array(sea_level)
        qc_flags = np.array(qc_flags)
        sea_level[qc_flags==5] = np.nan
        
        # Assign arrays to Dataset
        dataset['time'] = xr.DataArray(time, dims=['time'])
        dataset['sea_level'] = xr.DataArray(sea_level, dims=['time'])
        dataset['qc_flags'] = xr.DataArray(qc_flags, dims=['time'])
        
        # Assign local dataset to object-scope dataset
        return dataset