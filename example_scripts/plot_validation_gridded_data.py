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
fn_list = ["~/transfer/test_grid.nc", "~/transfer/test_grid.nc",]

# Filename for the output
fn_out = "/Users/dbyrne/transfer/surface_gridded_errors_{0}.png".format(run_name)

#%% General Plot Settings
var_name = "abs_diff_temperature"     # Variable name in analysis file to plot
                                  # If you used var modified to make gridded data
                                  # then this is where to select season etc.
save_plot = False

# Masking out grid cells that don't contain many points
min_points_in_average = 5
name_of_count_variable = "grid_N"

# Subplot axes settings
n_r = 2               # Number of subplot rows
n_c = 2               # Number of subplot columns
figsize = (10,5)      # Figure size
lonbounds = [-15,9.5] # Longitude bounds
latbounds = [45,64]   # Latitude bounds
subplot_padding = .5  # Amount of vertical and horizontal padding between plots
fig_pad = (.075, .075, .1, .1)  # Figure padding (left, top, right, bottom)
                                 # Leave some space on right for colorbar
# Scatter opts
marker_size = 3            # Marker size
cmap = 'bwr'               # Colormap for normal points
clim = (-1, 1)         # Color limits for normal points
discrete_cmap = True       # Discretize colormap
cmap_levels = 14

# Labels and Titles
fig_title = "SST Errors"                   # Whole figure title
title_fontsize = 13                        # Fontsize of title
title_fontweight = "bold"                  # Fontweight to use for title
dataset_names = ["CO9p0", "CO9p0", "CO9p0"]         # Names to use for labelling plots
subtitle_fontsize = 11                     # Fontsize for dataset subtitles
subtitle_fontweight = "normal"             # Fontweight for dataset subtitles

# PLOT SEASONS. Make sure n_r = 2 and n_c = 2
# If this option is true, only the first dataset will be plotted, with seasonal
# variables on each subplot. The season_suffixes will be added to var_name
# for each subplot panel.
plot_seasons = True
season_suffixes = ['DJF','MAM','JJA','SON']

#%% Read and plotdata

# Read all datasets into list
ds_list = [xr.open_dataset(dd) for dd in fn_list]
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
    
    
    # Select season if required
    if plot_seasons:
        ds = ds_list[0]
        var_ii = var_name + "_{0}".format(season_suffixes[ii])
        N_var = "{0}_{1}".format(name_of_count_variable, season_suffixes[ii])
        a_flat[ii].text(0.05, 1.02, season_suffixes[ii], 
                        transform=a_flat[ii].transAxes, fontweight='bold')
    else:
        ds = ds_list[ii]
        var_ii = var_name
        a_flat[ii].set_title(dataset_names[ii], fontsize = subtitle_fontsize,
                             fontweight = subtitle_fontweight)
        N_var = name_of_count_variable
        
    data = ds[var_ii].values
    count_var = ds[N_var]
    data[count_var<min_points_in_average] = np.nan
    
    # Scatter and set title
    pc = a_flat[ii].pcolormesh(ds.longitude, ds.latitude, data, cmap = cmap, 
                               vmin = clim[0], vmax = clim[1],)
    
    # Will we extend the colorbar for this dataset?
    extend_cbar.append( coast.plot_util.determine_colorbar_extension(data, clim[0], clim[1]) )

            
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
