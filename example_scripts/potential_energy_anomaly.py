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
#%%  Loading  data
#################################################

#  Loading AMM60 data if it is available
try:
    STOP
    config = "AMM60"
    dir_AMM60 = "/projectsa/COAsT/NEMO_example_data/AMM60/"
    fil_nam_dom = dir_AMM60 + "mesh_mask.nc"
    dir_AMM60 = "/projectsa/FASTNEt/kariho40/AMM60/RUNS/2010_2013/NO_DIFF/"
    fil_nam_AMM60 = dir_AMM60 + "AMM60_1d_20100704_20100708_grid_T.nc"
    fil_nam_AMM60 = dir_AMM60 + "AMM60_5d_20130814_20131012_grid_T.nc"
    config_t = "/work/jelt/GitHub/COAsT/config/example_nemo_grid_t.json"
    mon = "July"
    # mon = 'Feb'

    if mon == "July":
        fil_names_AMM60 = dir_AMM60 + "AMM60_1d_201007*_grid_T.nc"
        fil_names_AMM60 = "/projectsa/pycnmix/jelt/AMM60/AMM60_5d_20120801_20120831_grid_T.nc"
    elif mon == "Feb":
        fil_names_AMM60 = dir_AMM60 + "AMM60_1d_201002*_grid_T.nc"

    chunks = {
        "x_dim": 10,
        "y_dim": 10,
        "t_dim": 10,
    }  # Chunks are prescribed in the config json file, but can be adjusted while the data is lazy loaded.
    sci_t = coast.Gridded(
        fn_data=fil_names_AMM60, fn_domain=fil_nam_dom, config=config_t, multiple=True
    )
    sci_t.dataset = sci_t.dataset.chunk(chunks)


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

    sci_t = coast.Gridded(dn_files + fn_nemo_grid_t_dat, dn_files + fn_nemo_dom, config=config_t, multiple=True)

print("* Loaded ", config, " data")

#################################################
#%% Sort out bathymetry
try:
    if sci_nwes_t.dataset['bathymetry'] is None:
        print('Bathymetry variable missing')
        sci_nwes_t.dataset['bathymetry'] = sci_nwes_t.dataset.depth_0.where( sci_nwes_t.dataset.mask > 0 ).max(dim='z_dim')

    elif np.nanmax( sci_nwes_t.dataset['bathymetry'].values )<1:
        print('Bathymetry variable zero')
        sci_nwes_t.dataset['bathymetry'] = sci_nwes_t.dataset.depth_0.where( sci_nwes_t.dataset.mask > 0 ).max(dim='z_dim')
except:
        print('Problem sorting out bathymetry')

#################################################
#%% subset of data and domain ##
#################################################
# Pick out a North Sea subdomain

#print("* Extract North Sea subdomain")
#ind_sci = sci_t.subset_indices([51, -4], [62, 15])
#sci_nwes_t = sci_t.isel(y_dim=ind_sci[0], x_dim=ind_sci[1])  # nwes = northwest europe shelf
print("* Extract whole domain")
sci_nwes_t = sci_t

#%% Apply masks to temperature and salinity
if config == "AMM60":
    sci_nwes_t.dataset["temperature"] = sci_nwes_t.dataset.temperature.where(
        sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset["t_dim"].sizes) > 0
    )
    sci_nwes_t.dataset["salinity"] = sci_nwes_t.dataset.salinity.where(
        sci_nwes_t.dataset.mask.expand_dims(dim=sci_nwes_t.dataset["t_dim"].sizes) > 0
    )
    # Construct intermediate variables
    depth_0_4d = sci_nwes_t.dataset.depth_0.where( sci_nwes_t.dataset.mask > 0 ).to_masked_array()[np.newaxis, ...]
    e3_0_4d = sci_nwes_t.dataset.e3_0.where( sci_nwes_t.dataset.mask > 0 ).to_masked_array()[np.newaxis, ...]
    H = np.max(depth_0_4d, axis=1)
else:
    # Apply fake masks to temperature and salinity
    #sci_nwes_t.dataset["temperature_m"] = sci_nwes_t.dataset.temperature
    #sci_nwes_t.dataset["salinity_m"] = sci_nwes_t.dataset.salinity
    # Construct intermediate variables
    depth_0_4d = sci_nwes_t.dataset.depth_0.to_masked_array()[np.newaxis, ...]
    e3_0_4d = sci_nwes_t.dataset.e3_0.to_masked_array()[np.newaxis, ...]
    H = sci_nwes_t.dataset.bathymetry.to_masked_array()[np.newaxis, ...]

temp = sci_nwes_t.dataset["temperature"]
sal = sci_nwes_t.dataset["salinity"]



#%% Construct in-situ density
print("* Construct in-situ density")
sci_nwes_t.construct_density(eos="EOS10")


#%% Construct Potential Energy Anomaly
sci_nwes_t.construct_pea( eos="EOS10" )
pea = sci_nwes_t.dataset.pea


#%% North Sea Transect
tran_t = coast.TransectT(sci_nwes_t, (50, 2.5), (61, 2.5))

lat_sec = tran_t.data.latitude.expand_dims(dim={"z_dim": 51})
dep_sec = tran_t.data.depth_0
tem_sec = tran_t.data.temperature.mean(dim="t_dim")
sal_sec = tran_t.data.salinity.mean(dim="t_dim")
#rho_sec = tran_t.data.density.mean(dim="t_dim")
rho_sec = tran_t.data.density[0,:,:]
pea_sec = tran_t.data.pea[0,:]


#%%%% Transect figure
plt.figure()
plt.subplot(3,1,1)
plt.plot(tran_t.data.latitude, np.log10(pea_sec))
plt.title("PEA section (log10 J/m3)")
plt.xlim([50, 62])
plt.ylim([-1, 3])

plt.subplot(3,1,2)
plt.pcolormesh(lat_sec, dep_sec, rho_sec)
plt.title("density section")
plt.xlim([50, 62])
plt.ylim([0, 150])
plt.clim([1025, 1028])
plt.gca().invert_yaxis()
plt.colorbar()

plt.subplot(3,1,3)
plt.pcolormesh(lat_sec, dep_sec, sal_sec)
plt.title("sal section")
plt.xlim([50, 62])
plt.ylim([0, 150])
plt.clim([34, 35])
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


#%% Export as netcdf file
ofile = config + "_pea.nc"
sci_nwes_t.dataset.pea.to_netcdf(ofile, format="NETCDF4")
sci_nwes_t.dataset.bathymetry.to_netcdf(ofile, mode='a', format="NETCDF4")

#%%% Quick plot pea
plt.pcolormesh(sci_nwes_t.dataset.longitude, sci_nwes_t.dataset.latitude, np.log10(pea[-1,:,:]));plt.colorbar();
#plt.pcolormesh(sci_nwes_t.dataset.longitude, sci_nwes_t.dataset.latitude, np.log10(np.nanmean(sci_nwes_t.dataset.pea,axis=0)));plt.colorbar();
plt.title('Potential Energy Anomoly log10(J/m3)')
plt.show()


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


H = sci_nwes_t.dataset.bathymetry
lat = sci_nwes_t.dataset.latitude.squeeze()
lon = sci_nwes_t.dataset.longitude.squeeze()

# var = np.log10(np.nanmean(sci_nwes_t.dataset.pea[3,:,:],axis=0))  # make nan the land
var = np.log10(sci_nwes_t.dataset.pea[-1,:,:])  # make nan the land
# skipna = True --> ignore masked events when averaging
# skipna = False --> if once masked then mean is masked.

fig = plt.figure()
plt.rcParams["figure.figsize"] = (8.0, 8.0)

ax = fig.add_subplot(111)
cz = plt.contour(lon, lat, H, levels=[11, 50, 100, 200], colors=["k", "k", "k", "k"], linewidths=[1, 1, 1, 1])

plt.contourf(lon, lat, var, levels=np.arange(0, 3.0 + 0.25, 0.25), extend="both", cmap=new_cmap)
ax.set_facecolor("#bbbbbb")  # Set 'underneath' to grey. contourf plots nothing for bad values

# plt.xlim([-3, 11])
# plt.ylim([51, 62])
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


#%% Anthony code for PEA (counldn't get seawater working!!)
try:
    import xarray as xr
    import numpy as np
    import gsw
    from seawater import eos80 as sw

    def approx_depth_t(ds : xr.Dataset):
        depth_w = xr.zeros_like(ds.e3t)
        depth_w[dict(z_dim=slice(1,None))] = ds.e3t.cumsum(dim='z_dim').isel(z_dim=slice(0,-1))
        depth_w = depth_w.assign_coords({'k': ('z_dim', ds.z_dim.data)})
        e3w = depth_w.differentiate('k',edge_order=2)
        depth_t = xr.full_like(depth_w, np.nan)
        depth_t[dict(z_dim=0)] = e3w.isel(z_dim=0) * 0.5
        depth_t[dict(z_dim=slice(1,None))]  = e3w.cumsum(dim='z_dim').isel(z_dim=slice(1,None)) \
                                            + depth_t[dict(z_dim=0)]
        depth_t = depth_t.drop('k')
        return depth_t

    def pot_energy_anom(nemo_t: xr.Dataset, teos10=True):
        g = 9.81
        ds = nemo_t.dataset
        # get the approximate z coordinate (= -depth) for t-points
        z_t = -approx_depth_t(ds)
        if teos10==True:
            # Approx pressure from depth
            pressure = gsw.p_from_z( z_t, ds.latitude )
            # In situ density using TEOS-10, assumes absolute salinity and conservative temperature
            density = gsw.rho( ds.salinity, ds.temperature, pressure )
        else:
            # Approx pressure from depth
            pressure = sw.pres( -z_t, ds.latitude )
            # In situ density using EOS80, assumes practical salinity and temperature
            density = sw.dens( ds.salinity, ds.temperature, pressure )
        # get the water column thickness
        thickness = ds.e3t.sum(dim='z_dim',skipna=True).data
        # depth average density
        density_depthavg = (density*ds.e3t).sum(dim='z_dim') / thickness

        PEA = (g/thickness) * (( ( density_depthavg.data[:,np.newaxis,:,:] - density ) * z_t ) * ds.e3t ).sum(dim='z_dim')
        PEA.name='Potential Energy Anomaly'
except:
    print("Anthony's code is probably better. Uses seawater package")
