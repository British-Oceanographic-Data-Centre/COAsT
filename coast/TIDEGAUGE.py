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
    
    def read_gesla_to_dataset(self, fn_gesla, date_start=None, date_end=None):
        '''
        For reading from a single GESLA2 (Format version 3.0) file into an
        xarray dataset. Formatting according to Woodworth et al. (2017).
        Parameters
        ----------
        fn_gesla (str) : path to gesla tide gauge file
        date_start (datetime) : start date for returning data
        date_end (datetime) : end date for returning data
       
        Returns
        -------
        xarray.Dataset object.
        '''
        # Initialise empty dataset and lists
        dataset = xr.Dataset()
        lines = []
        time = []
        sea_level = []
        qc_flags = []
        # Open file and loop until EOF
        with open(fn_gesla) as file:
            line_count = 0
            for line in file:
                lines.append(line)
                
                # Header info
                if line_count==0:
                    pass
                elif line_count == 1: # Name of tide gauge site
                    dataset.attrs['site_name'] = line[12:]
                elif line_count == 4: # Latitude
                    dataset.attrs['latitude'] = float(line.split()[2])
                elif line_count == 5: # Longitude
                    dataset.attrs['longitude'] = float(line.split()[2])
                elif line_count == 9: # Time zone adjustment
                    dataset.attrs['time_zone_hours'] = float(line.split()[4])
                elif line_count == 12: # Precision
                    dataset.attrs['precision'] = float(line.split()[2])
                elif line_count == 13: # Null value
                    null_value = line.split()[3]
                    dataset.attrs['null_value'] = float(null_value)
                # Read all data. Date boundaries are set later.
                elif line_count>31:
                    working_line = line.split()
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
        
        # Attributes
        dataset.attrs['Source'] = 'Gesla 2 Database'
        
        # Assign local dataset to object-scope dataset
        self.dataset = dataset
        
        return #dataset
    
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