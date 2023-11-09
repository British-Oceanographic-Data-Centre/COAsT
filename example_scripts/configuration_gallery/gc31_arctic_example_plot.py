"""
Example of plotting Arctic data.
For plotting quivers over the Arctic see:
example_scripts/notebook_tutorials/runnable_notebooks/general/quiver_tutorial.ipynb

We can plot contours over the pole in the following way. This can be messy 
on the native grid because of wrapping over the pole. To work around this we 
re-project the data onto the NSIDC (National Snow and Ice Data Center) Polar 
Stereographic projection. As an example we plot the bathymetry contours.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import coast

root = "../../"
# Paths to a single or multiple data files.
dn_files = root + "example_files/"
fn_nemo_dat_t = dn_files + "HadGEM3-GC31-HH_hist_thetao.nc"
fn_nemo_config_t = root + "config/gc31_nemo_grid_t.json"

# Set path for domain file if required.
fn_nemo_dom = dn_files + "gc31_domain.nc"

# Define output filepath (optional: None or str)
fn_out = './gc31_arctic_plot.png'

# Read in multiyear data (This example uses NEMO data from a single file.)
nemo_data_t = coast.Gridded(fn_data=fn_nemo_dat_t,
                          fn_domain=fn_nemo_dom,
                          config=fn_nemo_config_t,
                          ).dataset

# Coarsen the data to make it run faster, also we don't need to plot the full
# resolution for such a wide area.
nemo_data_tc = nemo_data_t.coarsen(y_dim=10, x_dim=10).mean()

# Set things up for plotting North Pole stereographic projection

# Data projection
data_crs = ccrs.PlateCarree()
# Plot projection
mrc = ccrs.NorthPolarStereo(central_longitude=0.0)

# Data to plot
data_bathy = nemo_data_tc.bathymetry.values
data_temp = nemo_data_tc.temperature.isel(t_dim=0, z_dim=0)


figsize = (5, 5)  # Figure size
fig = plt.figure(figsize=figsize)
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.75], projection=mrc)
cax = fig.add_axes([0.3, 0.96, 0.4, 0.01])

ax1.add_feature(cfeature.LAND, zorder=105)
ax1.gridlines()
ax1.set_extent([-180, 180, 70, 90], crs=data_crs)
coast._utils.plot_util.set_circle(ax1)

# We use to a function to re-project the data for plotting contours over the pole.
cs1 = coast._utils.plot_util.plot_polar_contour(nemo_data_tc.longitude.values,
                        nemo_data_tc.latitude.values,
                        data_bathy, ax1, levels=6, colors='k', zorder=101)
cs2 = ax1.pcolormesh(nemo_data_tc.longitude.values, nemo_data_tc.latitude.values,
                     data_temp, transform=data_crs, vmin=-2, vmax=8)

fig.colorbar(cs2, cax=cax, orientation='horizontal')
cax.set_xlabel(r'SST ($^{\circ}$C)')

fig.savefig(fn_out, dpi=120)
