#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:30:21 2020

You might scrpae tidal highs and lows from a website such as

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
import matplotlib.pyplot as plt



if(1):
    print('load and plot HLW data')
    filnam = 'example_files/Gladstone_2020-10_HLW.txt'
    date_start = np.datetime64('2020-10-13')
    date_end = np.datetime64('2020-10-14')
    tg = coast.TIDEGAUGE()
    tg.dataset = tg.read_HLW_to_xarray(filnam, date_start, date_end)
    # Exaple plot
    tg.dataset.plot.scatter(x="time", y="sea_level")


if(1):
    print("Calculate some basic statistics")
    print(f"stats: mean {tg.time_mean('sea_level')}")
    print(f"stats: std {tg.time_std('sea_level')}")

