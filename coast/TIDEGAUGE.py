import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import pandas as pd

class TIDEGAUGE():
    '''
    An object for reading, storing and manipulating tide gauge data.
    Data is stored for just a single tide gauge.
    Data is kept in the objects xarray.Dataset().
    '''  
    
    def __init__(self):
        self.dataset = None
        return
    
    def read_gesla_all(self):
        return
    
    def read_gesla_radius(self):
        return
    
    def read_gesla_list(self, file_list, date_start, date_end):
        
        dataset_list = []
        site_names = []
        
        for fn in file_list:
            tmp_dataset = self.read_gesla_to_dataset(fn, date_start,
                                date_end)
            dataset_list.append()
            
        # Establish commonalities between datasets
        
        
        return
    
    def read_gesla_v3(self, fn_gesla, date_start=None, date_end=None):
        '''
        For reading from a single GESLA2 (Format version 3.0) file into an
        xarray dataset. Formatting according to Woodworth et al. (2017).
        Website: https://www.gesla.org/
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
            header_dict = self.read_gesla_header_v3(fn_gesla)
            dataset = self.read_gesla_data_v3(fn_gesla, date_start, date_end)
        except:
            raise Exception('Problem reading GESLA file: ' + fn_gesla)
        # Attributes
        dataset.attrs = header_dict
        
        # Assign local dataset to object-scope dataset
        self.dataset = dataset
        
        return
    
    def read_gesla_header_v3(self, fn_gesla):
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
    
    def read_gesla_data_v3(self, fn_gesla, date_start=None, date_end=None,
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
    
    def quick_plot(self, date_start=None, date_end=None, qc_colors=True,
                   plot_line = False):
        '''
        Quick plot of time series stored within object's dataset
        Parameters
        ----------
        date_start (datetime) : Start date for plotting
        date_end (datetime) : End date for plotting
        qc_colors (bool) : If true, markers are coloured according to qc values
        plot_line (bool) : If true, draw line between markers
       
        Returns
        -------
        matplotlib figure and axes objects
        '''
        
        
        # Numpyify data
        x = np.array(self.dataset.time)
        y = np.array(self.dataset.sea_level)
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
            plt.scatter(x,y)
            plt.grid()
            plt.xticks(rotation=65)
            
        if plot_line:
            plt.plot(x,y, c=[0.5,0.5,0.5], linestyle='--', linewidth=0.5)
        
        return fig, ax