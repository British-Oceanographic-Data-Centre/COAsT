'''
Plot up surface or bottom (or any fixed level) errors from a profile object
with no z_dim (vertical dimension). Provide an array of netcdf files and 
mess with the options to get a figure you like.

You can define how many rows and columns the plot will have. This script will
plot the provided list of netcdf datasets from left to right and top to bottom.

A colorbar will be placed right of the figure.
'''

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/dbyrne/code/COAsT')
import coast
import pandas as pd

#%% File settings
run_name = "test"

# List of analysis output files. Profiles from each will be plotted
# on each axis of the plot
fn_list = ["/Users/dbyrne/transfer/surface_data_gridded_test.nc",
           "/Users/dbyrne/transfer/surface_data_gridded_test.nc",]

# Filename for the output
fn_out = "/Users/dbyrne/transfer/surface_gridded_errors_{0}.png".format(run_name)

#%% General Plot Settings
var_name = "diff_temperature"     # Variable name in analysis file to plot
                                  # If you used var modified to make gridded data
                                  # then this is where to select season etc.
save_plot = False

# Subplot axes settings
n_r = 1               # Number of subplot rows
n_c = 2               # Number of subplot columns
figsize = (10,5)      # Figure size
lonbounds = [-18,9.5] # Longitude bounds
latbounds = [45,64]   # Latitude bounds
subplot_padding = .5  # Amount of vertical and horizontal padding between plots
fig_pad = (.075, .075, .1, .1)  # Figure padding (left, top, right, bottom)
                                 # Leave some space on right for colorbar
# Scatter opts
marker_size = 3            # Marker size
cmap = 'bwr'               # Colormap for normal points
clim = (-.35, .35)         # Color limits for normal points
discrete_cmap = True       # Discretize colormap
cmap_levels = 13

# Labels and Titles
fig_title = "SST Errors"                   # Whole figure title
title_fontsize = 13                        # Fontsize of title
title_fontweight = "bold"                  # Fontweight to use for title
dataset_names = ["CO9p0", "CO9p0", "CO9p0"]         # Names to use for labelling plots
subtitle_fontsize = 11                     # Fontsize for dataset subtitles
subtitle_fontweight = "normal"             # Fontweight for dataset subtitles


#%% Read and plotdata

# Read all datasets into list
ds_list = [xr.open_dataset(dd)[var_name] for dd in fn_list]
n_ds = len(ds_list)
n_ax = n_r*n_c

# Create plot and flatten axis array
f,a = coast.plot_util.create_geo_subplots(lonbounds, latbounds, n_r, n_c, figsize = figsize)
a_flat = a.flatten()

# Dicretize colormap maybe
if discrete_cmap:
    cmap = plt.cm.get_cmap(cmap, cmap_levels)
    
# Determine if we will extend the colormap or not
extend_cbar = []

# Loop over dataset
for ii in range(n_ax):
    ur_index = np.unravel_index(ii, (n_r, n_c))
    
    # If we are not differencing datasets
    ds = ds_list[ii]
    
    # Select season if required
        
    # Scatter and set title
    pc = a_flat[ii].pcolormesh(ds.longitude, ds.latitude, ds, cmap = cmap, 
                               vmin = clim[0], vmax = clim[1],)
    a_flat[ii].set_title(dataset_names[ii], fontsize = subtitle_fontsize,
                         fontweight = subtitle_fontweight)
    
    # Will we extend the colorbar for this dataset?
    extend_cbar.append( coast.plot_util.determine_colorbar_extension(ds, clim[0], clim[1]) )

            
# Set Figure title
f.suptitle(fig_title, fontsize = title_fontsize, fontweight = title_fontweight)


# Set tight figure layout
f.tight_layout(w_pad = subplot_padding, h_pad= subplot_padding)
f.subplots_adjust(left   = (fig_pad[0]), 
                  bottom = (fig_pad[1]),
                  right  = (1-fig_pad[2]),
                  top    = (1-fig_pad[3]))

# Handle colorbar -- will we extend it?
if 'both' in extend_cbar:
    extend = 'both'
elif 'max' in extend_cbar and 'min' in extend_cbar:
    extend = 'both'
elif 'max' in extend_cbar:
    extend = 'max'
elif 'min' in extend_cbar:
    extend = 'min'
else:
    extend = 'neither'
cbar_ax = f.add_axes([(1 - fig_pad[2] + fig_pad[2]*0.15), .15, .025, .7])
f.colorbar(pc, cax = cbar_ax, extend=extend)

# Save plot maybe
if save_plot: 
    f.savefig(fn_out)
