#!/usr/bin/env python3
"""
Tutorial for processing tabulated tide gauge data

Created on Mon Oct 12 22:30:21 2020

You might scrape tidal highs and lows from a website such as

<a title="NTSLF tidal predictions"
href="https://www.ntslf.org/tides/tidepred?port=Liverpool">
<img alt="NTSLF tidal predictions"
src="https://www.ntslf.org/files/ntslf_php/plottide.php?port=Liverpool" height="200" width="290" /></a>

and format them into a csv file:

LIVERPOOL (GLADSTONE DOCK)    TZ: UT(GMT)/BST     Units: METRES    Datum: Chart Datum
01/10/2020  06:29    1.65
01/10/2020  11:54    9.01
01/10/2020  18:36    1.87
...

The following demonstration would allow you to pass these data.

@author: jeff
"""
import coast
import numpy as np
import xarray as xr


#%% Load and plot High and Low Water data
print('load and plot HLW data')
filnam = 'example_files/Gladstone_2020-10_HLW.txt'

# Set the start and end dates
date_start = np.datetime64('2020-10-12 23:59')
date_end = np.datetime64('2020-10-14 00:01')

# Initiate a TIDEGAUGE object, if a filename is passed it assumes it is a GESLA type object
tg = coast.TIDEGAUGE()
# specify the data read as a High Low Water dataset
tg.dataset = tg.read_HLW_to_xarray(filnam, date_start, date_end)
# Show dataset. If timezone is specified then it is presented as requested, otherwise uses UTC
print("Try the TIDEGAUGE.show() method:")
tg.show(timezone = 'Europe/London')
# Do a basic plot of these points
tg.dataset.plot.scatter(x="time", y="sea_level")

#%%

# There is a method to locate HLW events around an approximate date and time
# First state the time of interest
time_guess = np.datetime64('2020-10-13 12:48')
# Then recover all the HLW events in a +/- window, of specified size (iteger hrs)
# The default winsize = 2 (hrs)
HLW = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='window', winsize=24 )

# Alternatively recover the closest HLW event to the input timestamp
HLW = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_1' )

# Or the nearest two events to the input timestamp
HLW = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_2' )

# Extract the Low Tide value
print("Try the TIDEGAUGE.get_tidetabletimes() methods:")
print('LT:', HLW[ HLW.argmin() ].values, 'm at', HLW[ HLW.argmin() ].time.values )

# Extract the High Tide value
print('HT:', HLW[ HLW.argmax() ].values, 'm at', HLW[ HLW.argmax() ].time.values )

# Or use the the nearest High Tide method to get High Tide
HT = tg.get_tidetabletimes( np.datetime64('2020-10-13 12:48'), method='nearest_HW' )
print('HT:', HT.values, 'm at', HT.time.values )
