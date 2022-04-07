"""
For plotting analysis data from a netcdf file created using COAsT.Profile.mask_means().
This will plot multiple datasets onto a set of subplots. Each subplot is for
a different averaging region.

At the top of this script, you can set the paths to the netcdf files to plot
and where to save. If you have multiple model runs to plot, provide a list
of file paths (strings).

Below this section are a bunch of parameters you can set, with explanations in
comments. Edit this as much as you like or even go into the plotting code below.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#%% File settings
run_name = "test"

# List of analysis output files. Profiles from each will be plotted
# on each axis of the plot
fn_list = ["/Users/dbyrne/transfer/mask_means_daily_test.nc", "/Users/dbyrne/transfer/mask_means_daily_test.nc"]

# Filename for the output
fn_out = "/Users/dbyrne/transfer/regional_means_{0}.png".format(run_name)

#%% General Plot Settings
region_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Region indices (in analysis) to plot
region_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]  # Region names, will be used for titles in plot
var_name = "profile_average_abs_diff_temperature"  # Variable name in analysis file to plot
plot_zero_line = True  # Plot a black vertical line at x = 0
plot_mean_depth = False  # Plot the mean bathymetric depth. Make sure 'bathymetry' is in the analysis dataset
save_plot = False  # Boolean to save plot or not

ref_depth = np.concatenate((np.arange(1, 100, 2), np.arange(100, 300, 5), np.arange(300, 1000, 50)))  # Data depths

# Subplot axes settings
n_r = 2  # Number of subplot rows
n_c = 5  # Number of subplot columns
figsize = (7, 7)  # Figure size
sharey = True  # Align y axes
sharex = False  # Align x axes
subplot_padding = 0.5  # Amount of vertical and horizontal padding between plots
fig_pad = (0.075, 0.075, 0.075, 0.1)  # Whole figure padding as % (left, top, right, bottom)
max_depth = 200  # Maximum plot depth

# Legend settings
legend_str = ["CO9p0", "CO9p0_2"]  # List of strings to use in legend (match with fn_list ordering)
legend_index = 9  # Axis index to put legend (flattened index, start from 0).
# Good to place in an empty subplot
legend_pos = "upper right"  # Position for legend, using matplitlib legend string
legend_fontsize = 9

# Labels and Titles
xlabel = "Absolute Error (degC)"  # Xlabel string
xlabelpos = (figsize[0] / 2, 0)  # (x,y) position of xlabel
ylabel = "Depth (m)"  # Ylabel string
ylabelpos = (figsize[1] / 2, 0)  # (x,y) position of ylabel
fig_title = "Regional MAE || All Seasons"  # Whole figure title
label_fontsize = 11  # Fontsize of all labels
label_fontweight = "normal"  # Fontweight to use for labels and subtitles
title_fontsize = 13  # Fontsize of title
title_fontweight = "bold"  # Fontweight to use for title


#%% SCRIPT: READ AND PLOT DATA

# Read all datasets into list
ds_list = [xr.open_dataset(dd) for dd in fn_list]
n_ds = len(ds_list)
n_reg = len(region_ind)
n_ax = n_r * n_c

# Create plot and flatten axis array
f, a = plt.subplots(n_r, n_c, figsize=figsize, sharex=sharex, sharey=sharey)
a_flat = a.flatten()

# Loop over regions
for ii in range(n_ax):

    if ii >= n_reg:
        a_flat[ii].axis("off")
        continue

    # Get the index of this region
    index = region_ind[ii]

    # Loop over datasets and plot their variable
    p = []
    for pp in range(n_ds):
        ds = ds_list[pp]
        p.append(a_flat[ii].plot(ds[var_name][index], ref_depth)[0])

    # Do some plot things
    a_flat[ii].set_title(region_names[ii])
    a_flat[ii].grid()
    a_flat[ii].set_ylim(0, max_depth)

    # Plot fixed lines at 0 and mean depth
    if plot_zero_line:
        a_flat[ii].plot([0, 0], [0, max_depth], c="k", linewidth=1, linestyle="-")
    if plot_mean_depth:
        a_flat[ii].plot()

    # Invert y axis
    a_flat[ii].invert_yaxis()

# Make legend
a_flat[legend_index].legend(p, legend_str, fontsize=legend_fontsize)

# Set Figure title
f.suptitle(fig_title, fontsize=title_fontsize, fontweight=title_fontweight)

# Set x and y labels
f.text(
    xlabelpos[0],
    xlabelpos[1],
    xlabel,
    va="center",
    rotation="horizontal",
    fontweight=label_fontweight,
    fontsize=label_fontsize,
)

# Set tight figure layout
f.tight_layout(w_pad=subplot_padding, h_pad=subplot_padding)
f.subplots_adjust(left=(fig_pad[0]), bottom=(fig_pad[1]), right=(1 - fig_pad[2]), top=(1 - fig_pad[3]))

# Save plot maybe
if save_plot:
    f.savefig(fn_out)
