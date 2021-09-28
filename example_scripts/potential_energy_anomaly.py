"""
potential_energy_anomaly.py

Demonstration of pycnocline depth and thickness diagnostics.
The first and second depth moments of stratification are computed as proxies
for pycnocline depth and thickness, suitable for a nearly two-layer fluid.


"""

#%%
import coast
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # colormap fiddling

#################################################
#%%  Define   cnstants
#################################################
g = 9.81


#################################################
#%%  Define methods
#################################################
def quick_plot(var: xr.DataArray = None):
    """

    Map plot for PEA.

    Parameters
    ----------
    var : xr.DataArray, optional
        ignored: PEA is plotted.

    Returns
    -------
    None.

    Example Usage
    -------------
    pea.quick_plot()

    """
    import matplotlib.pyplot as plt

  

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.pcolormesh(var.longitude.squeeze(), var.latitude.squeeze(), var.data[0,:,:].squeeze())
    #               var.mean(dim = 't_dim') )
    # plt.contourf( self.dataset.longitude.squeeze(),
    #               self.dataset.latitude.squeeze(),
    #               var.mean(dim = 't_dim'), levels=(0,10,20,30,40) )
    title_str = (
        var.time[0].dt.strftime("%d %b %Y: ").values
        + var.attrs["standard_name"]
        + " ("
        + var.attrs["units"]
        + ")"
    )
    plt.title(title_str)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    #plt.clim([0, 50])
    plt.colorbar()
    plt.show()
    return fig, ax  # TODO if var_lst is empty this will cause an error

#################################################
#%%  Loading  data
#################################################

#  Loading AMM60 data if it is available
try:
    config = "AMM60"
    dir_AMM60 = "/projectsa/COAsT/NEMO_example_data/AMM60/"
    fil_nam_AMM60 = "AMM60_1d_20100704_20100708_grid_T.nc"
    config_t = "/work/jelt/GitHub/COAsT/config/example_nemo_grid_t.json"
    config_w = "/work/jelt/GitHub/COAsT/config/example_nemo_grid_w.json"
    mon = "July"
    # mon = 'Feb'

    if mon == "July":
        fil_names_AMM60 = "AMM60_1d_201007*_grid_T.nc"
    elif mon == "Feb":
        fil_names_AMM60 = "AMM60_1d_201002*_grid_T.nc"

    chunks = {
        "x_dim": 10,
        "y_dim": 10,
        "t_dim": 10,
    }  # Chunks are prescribed in the config json file, but can be adjusted while the data is lazy loaded.
    sci_t = coast.Gridded(
        fn_data=dir_AMM60 + fil_names_AMM60, fn_domain=dir_AMM60 + "mesh_mask.nc", config=config_t, multiple=True
    )
    sci_t.dataset = sci_t.dataset.chunk(chunks)

    # create an empty w-grid object, to store stratification
    sci_w = coast.Gridded(fn_domain=dir_AMM60 + "mesh_mask.nc", config=config_w)

# OR load in AMM7 example data
except:
    config = "AMM7"
    dn_files = "./example_files/"

    if not os.path.isdir(dn_files):
        print("please go download the examples file from https://linkedsystems.uk/erddap/files/COAsT_example_files/")
        dn_files = input("what is the path to the example files:\n")
        if not os.path.isdir(dn_files):
            print(f"location f{dn_files} cannot be found")

    dn_fig = "unit_testing/figures/"
    fn_nemo_grid_t_dat = "nemo_data_T_grid_Aug2015.nc"
    fn_nemo_dom = "coast_example_nemo_domain.nc"
    config_t = "config/example_nemo_grid_t.json"
    config_w = "config/example_nemo_grid_w.json"

    sci_t = coast.Gridded(dn_files + fn_nemo_grid_t_dat, dn_files + fn_nemo_dom, config=config_t, multiple=True)

    # create an empty w-grid object, to store stratification
    sci_w = coast.Gridded(fn_domain=dn_files + fn_nemo_dom, config=config_w)
print("* Loaded ", config, " data")

#################################################
#%% subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
print("* Extract North Sea subdomain")
ind_sci = sci_t.subset_indices([51, -4], [62, 15])
sci_nwes_t = sci_t.isel(y_dim=ind_sci[0], x_dim=ind_sci[1])  # nwes = northwest europe shelf
ind_sci = sci_w.subset_indices([51, -4], [62, 15])
sci_nwes_w = sci_w.isel(y_dim=ind_sci[0], x_dim=ind_sci[1])  # nwes = northwest europe shelf

#%% Apply masks to temperature and salinity
if config == "AMM60":
    sci_nwes_t.dataset["temperature_m"] = sci_nwes_t.dataset.temperature.where(
        sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset["t_dim"].sizes) > 0
    )
    sci_nwes_t.dataset["salinity_m"] = sci_nwes_t.dataset.salinity.where(
        sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset["t_dim"].sizes) > 0
    )

else:
    # Apply fake masks to temperature and salinity
    sci_nwes_t.dataset["temperature_m"] = sci_nwes_t.dataset.temperature
    sci_nwes_t.dataset["salinity_m"] = sci_nwes_t.dataset.salinity


#%% Construct in-situ density and pea
print("* Construct in-situ density and stratification")
sci_nwes_t.construct_density(eos="EOS10")


sci_nwes_t.construct_pea()
pea = sci_nwes_t.dataset.pea



#%%%%


plt.pcolormesh(np.log10(np.nanmean(sci_nwes_t.dataset.pea,axis=0)));plt.colorbar();
plt.title('Potential Energy Anomoly log10(J/m3)')
plt.show()


#%% Export as netcdf file
ofile = "pea.nc"
sci_nwes_t.dataset.pea.to_netcdf(ofile, format="NETCDF4")


#%% Map pretty plots of North Sea PEA 
print("* Map pretty plots of North Sea Potential energy anomaly")


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


cmap = plt.get_cmap("BrBG_r")
new_cmap = truncate_colormap(cmap, 0.2, 0.8)
# new_cmap.set_bad(color = '#bbbbbb') # light grey
new_cmap.set_under(color="w")  # white.
# It would be nice to plot the unstratified regions different to the land.


H = sci_nwes_t.dataset.depth_0[-1, :, :].squeeze()
lat = sci_nwes_t.dataset.latitude.squeeze()
lon = sci_nwes_t.dataset.longitude.squeeze()

var = np.log10(np.nanmean(sci_nwes_t.dataset.pea,axis=0))  # make nan the land
# skipna = True --> ignore masked events when averaging
# skipna = False --> if once masked then mean is masked.

fig = plt.figure()
plt.rcParams["figure.figsize"] = (8.0, 8.0)

ax = fig.add_subplot(111)
cz = plt.contour(lon, lat, H, levels=[11, 50, 100, 200], colors=["k", "k", "k", "k"], linewidths=[1, 1, 1, 1])

plt.contourf(lon, lat, var, levels=np.arange(0, 3.0 + 0.5, 0.5), extend="both", cmap=new_cmap)
ax.set_facecolor("#bbbbbb")  # Set 'underneath' to grey. contourf plots nothing for bad values

plt.xlim([-3, 11])
plt.ylim([51, 62])
plt.colorbar()


lines = [
    cz.collections[i] for i in range(1, len(cz.collections))
]  # [ cz.collections[1], cz.collections[2], cz.collections[-1] ]
labels = [str(int(cz.levels[i])) + "m" for i in range(1, len(cz.levels))]
# labels = ['80m','200m','800m']

plt.legend(lines, labels, loc="lower right")

# I expect to see RuntimeWarnings in this block
title_str = (
    sci_nwes_t.dataset["time"].mean(dim="t_dim").dt.strftime("%b %Y: ").values
    + sci_nwes_t.dataset.pea.attrs['standard name']
    + " ("
    + sci_nwes_t.dataset.pea.attrs['units']
    + ")"
)
plt.title(title_str)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()

# fig.savefig(fig_dir+'strat_1st_mom.png', dpi=120)
