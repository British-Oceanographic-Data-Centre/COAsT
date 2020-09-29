## ANChor_plots_of_NSea_pycnocline.py
"""

DEV_jelt/NEMO_diag/ANChor
This needs to move to the above
"""

#%%
import coast
import numpy as np
import xarray as xr
#import dask
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as colors # colormap fiddling

#################################################
#%%  Loading and initialising methods ##
#################################################
    
dir_nam = "/projectsa/anchor/NEMO/AMM60/"
fil_nam = "AMM60_1h_20100818_20100822_NorthSea.nc"
dom_nam = "/projectsa/anchor/NEMO/AMM60/mesh_mask.nc"
        
dir_nam = '/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/'
fil_nam = 'AMM60_1h_20120204_20120208_NorthSea.nc'

sci_w = coast.NEMO(dir_nam + fil_nam, 
                 dom_nam, grid_ref='w-grid', multiple=True)


#################################################
#%% subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
ind_sci = sci_w.subset_indices([51,-4], [60,15])
sci_nwes_w = sci_w.isel(y_dim=ind_sci[0], x_dim=ind_sci[1]) #nwes = northwest europe shelf

#sci_nwes_w.dataset.wo.isel(depthw=5).isel(t_dim=1).plot()
#%% Apply masks to temperature and salimity
#sci_nwes_w.dataset['temperature_m'] = sci_nwes_w.dataset.temperature.where( sci_nwes_w.dataset.mask.expand_dims(dim=sci_nwes_w.dataset['t_dim'].sizes) > 0) 
#sci_nwes_w.dataset['salinity_m'] = sci_nwes_w.dataset.salinity.where( sci_nwes_w.dataset.mask.expand_dims(dim=sci_nwes_w.dataset['t_dim'].sizes) > 0) 


   
#%% Transect Method

#tran = coast.Transect( (54,-15), (56,-12), nemo_f, nemo_t, nemo_u, nemo_v )
tran = coast.Transect( (51, 2.5), (60, 2.5), sci_nwes_w )

#print( tran.data_F.latitude.expand_dims(dim={'z_dim':51}).shape, tran.data_F.depth_0.shape, tran.data_F.temperature_m.shape )

lat_sec = tran.data_F.latitude.expand_dims(dim={'z_dim':51})
dep_sec = tran.data_F.depth_0
wo_sec = tran.data_F.wo
#wo_sec = tran.data_F.wo.mean(dim='t_dim')



#%% Make map and profile plots
#################################################
for i in range(2):
    for lat0 in [54, 57]:
        if lat0==54:
            sig0=10
            lon0=5 # depth level for maps
        if lat0==57:
            sig0=40
            lon0=2
        [JJ,II] = sci_nwes_w.find_j_i( lat= lat0, lon= lon0)
        # short cuts for variable names
        lon =  sci_nwes_w.dataset.longitude.squeeze()
        lat =  sci_nwes_w.dataset.latitude.squeeze()
        dep =  sci_nwes_w.dataset.depth_0[:,:,:]
        
        fig = plt.figure()
        plt.rcParams['figure.figsize'] = 8,8
        
        fig = plt.figure()
        plt.rcParams['figure.figsize'] = 8,8
        plt.pcolormesh(lon, lat, sci_nwes_w.dataset.wo[i,sig0,:,:]*3600*24, 
                       shading='auto', cmap='seismic')
        plt.plot(lon[JJ,II], lat[JJ,II], 'r+')
        plt.title(f"t={str(i)}: w-vel (m/day) at level {sig0}")
        plt.clim([-5,5])
        plt.colorbar()
        fig.savefig(f"w_map_sig{sig0}_{str(i).zfill(3)}.png")
        
        fig = plt.figure()
        plt.rcParams['figure.figsize'] = 8,8
        
        plt.subplot(1,2,1)
        plt.plot(sci_nwes_w.dataset.wo[i,:,JJ,II]*3600*24,  dep[:,JJ,II], '+')
        plt.title(f"w-vel (m/day) at ({lat0}N,{lon0}E)")
        plt.xlim([-15,15])
        plt.ylabel('depth (m)')
        plt.gca().invert_yaxis()
        
        plt.subplot(1,2,2)
        plt.plot(sci_nwes_w.dataset.avm[i,:,JJ,II]*1e3,  dep[:,JJ,II], '+')
        plt.title(f"t={str(i)}: avm*1E3")
        plt.ylabel('depth (m)')
        plt.xlim([0,40])
        plt.gca().invert_yaxis()
        fig.savefig(f"w_prof_{lat0}N_{str(i).zfill(3)}.png")
        
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = 8,8
    plt.pcolormesh(  lat_sec, dep_sec, wo_sec.isel(t_dim=i)*3600*24,
        shading='auto', cmap='seismic')
    plt.colorbar()
    plt.title(f"t={str(i)}: w-vel section (m/day)")
    plt.xlim([51,60])
    plt.ylim([0,150])
    plt.xlabel('latitude')
    plt.ylabel('depth (m)')
    plt.clim([-20,20])
    plt.gca().invert_yaxis()
    fig.savefig(f"w_section_t_{str(i).zfill(3)}.png")
        
    plt.close('all')


#%% Plot sections
fig = plt.figure()
plt.rcParams['figure.figsize'] = 8,8

plt.subplot(1,1,1)

plt.pcolormesh(  lat_sec, dep_sec, wo_sec.mean(dim='t_dim')*3600*24,
		shading='auto', cmap='seismic')
plt.title('w-vel t-mean section')
plt.xlim([51,60])
plt.ylim([0,150])
plt.clim([-20, 20])
plt.xlabel('latitude')
plt.ylabel('depth (m)')
plt.gca().invert_yaxis()
plt.colorbar()
fig.savefig(f"w_section_tmean.png")



