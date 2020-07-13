## ANChor_plots_of_NSea_pycnocline.py
"""

DEV_jelt/NEMO_diag/ANChor
This needs to move to the above
"""

#%%
import coast
import numpy as np
import xarray as xr
import dask
#import matplotlib.pyplot as plt


#%%

#dir = "example_files/"
dir = "/Users/jeff/downloads/"

fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
#fn_altimetry = 'COAsT_example_altimetry_data.nc'

#%%

# Copy domain subsetting method to NEMO class
if(1):
# Add subset method to NEMO class
    def subset_indices(self, start: tuple, end: tuple, grid_ref: str = 'T') -> tuple:
        """
        based off transect_indices, this method looks to return all indices between the given points.
        This results in a 'box' (Quadrilateral) of indices.
        consequently the returned lists may have different lengths.

        :param start: A lat/lon pair
        :param end: A lat/lon pair
        :param grid_ref: The gphi/glam version a user wishes to search over
        :return: list of y indices, list of x indices,
        """
        assert isinstance(grid_ref, str) and grid_ref.upper() in ("T", "V", "U", "F"), \
            "grid_ref should be either \"T\", \"V\", \"U\", \"F\""

        letter = grid_ref.lower()

        [j1, i1] = self.find_j_i(start[0], start[1], letter)  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1], letter)  # lat , lon

        return list(np.arange(j1, j2+1)), list(np.arange(i1, i2+1))
    

    def find_j_i(self, lat: int, lon: int, grid_ref: str):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12, t)

        :param lat: latitude
        :param lon: longitude
        :param grid_ref: the gphi/glam version a user wishes to search over
        :return: the y and x coordinates for the given grid_ref variable within the domain file
        """

        internal_lat = "nav_lat"
        internal_lon = "nav_lon"
        dist2 = xr.ufuncs.square(self.dataset[internal_lat] - lat) + xr.ufuncs.square(self.dataset[internal_lon] - lon)
        [ y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]   
    
    setattr(coast.NEMO, 'subset_indices', subset_indices)
    setattr(coast.NEMO, 'find_j_i', find_j_i)

#%%

#################################################
## Loading and initialising methods ##
#################################################

#sci = coast.NEMO(dir + fn_nemo_dat)
#dom = coast.DOMAIN(dir + fn_nemo_dom)
##alt = coast.ALTIMETRY(dir + fn_altimetry)

#sci = coast.NEMO("/projectsa/COAsT/ind_1h_20180101_20160109_shelftmb_grid_T.nc")
#dom = coast.DOMAIN("/projectsa/accord/BoBEAS/INPUTS/domain_cfg.nc")

#sci = coast.NEMO("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_20100130_20100203_grid_T.nc")
#sci = coast.NEMO("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_20100704_20100708_grid_T.nc") 
sci = coast.NEMO("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_201007*_grid_T.nc", multiple=True) 
dom = coast.DOMAIN("/projectsa/FASTNEt/jelt/AMM60/mesh_mask.nc")
#%%
#ds = xr.open_mfdataset("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_201007*_grid_T.nc")
#sci_load_ds = coast.NEMO()
#sci_load_ds.load_dataset(ds)
#sci = sci_load_ds
#################################################
## subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
ind = dom.subset_indices([51,-4], [62,15])

#ind = dom.subset_indices([10,90], [15,95]) # Andermann Shelf

dom_nwes = dom.isel(y_dim=ind[0], x_dim=ind[1]) #nwes = northwest europe shelf


#%%

ind_sci = sci.subset_indices([51,-4], [62,15])

sci_nwes = sci.isel(y_dim=ind_sci[0], x_dim=ind_sci[1]) #nwes = northwest europe shelf



#%%

#################################################
## Create Diagnostics object
#################################################
IT_obj = coast.DIAGNOSTICS(sci_nwes, dom_nwes)

# Construct stratification
IT_obj.get_stratification( sci_nwes.dataset.thetao ) # --> self.strat

# Construct pycnocline variables: depth and thickness
IT_obj.get_pyc_vars() # --> self.zd and self.zt

#%%

#################################################
## Make Plots
#################################################
import matplotlib.pyplot as plt
import matplotlib.colors as colors # colormap fiddling

#%%

[JJ,II] = sci_nwes.find_j_i( lat= 55, lon= 5, grid_ref='t')

# short cuts for variable names
lon =  IT_obj.domain.dataset.glamt.squeeze()
lat =  IT_obj.domain.dataset.gphit.squeeze()
dep =  IT_obj.domain.depth_t[:,:,:]

fig = plt.figure()
plt.rcParams['figure.figsize'] = 8,8

fig = plt.figure()
plt.rcParams['figure.figsize'] = 8,8
plt.pcolormesh(lon, lat, IT_obj.strat[0,10,:,:])
plt.title('stratification')
plt.colorbar()
plt.plot(lon[JJ,II], lat[JJ,II], 'k+')
plt.show()

plt.plot( IT_obj.strat[0,:,JJ,II], dep[:,JJ,II], '+')
plt.title('stratification')
plt.ylabel('depth (m)')
plt.gca().invert_yaxis()
plt.show()

#plt.plot(sci_nwes.dataset.votemper[0,:,100,60],'+'); plt.title('temperature'); plt.show()
plt.plot(sci_nwes.dataset.thetao[0,:,JJ,II],  dep[:,JJ,II], '+')
plt.title('temperature (decC)')
plt.ylabel('depth (m)')
plt.gca().invert_yaxis()
plt.show()

plt.pcolormesh( lon, lat, IT_obj.zd[0,:,:])
plt.title('pycnocline depth (m)')
plt.clim([0, 50])
plt.colorbar()
plt.plot(lon[JJ,II], lat[JJ,II], 'k+')
plt.show()

plt.pcolormesh( lon, lat, IT_obj.zt[0,:,:])
plt.title('pycnocline thickness (m)')
plt.clim([0, 50])
plt.colorbar()
plt.plot(lon[JJ,II], lat[JJ,II], 'k+')
plt.show()

#%%

# Extract a depth section of data
yi,xi,line_len = dom_nwes.transect_indices([51,3],[62,3], grid_ref='t')
# Extract the variable
data_sec = sci_nwes.get_subset_as_xarray("thetao",xi,yi)
# Extract the depth section
#_, dom_nwes.glamt_4d = xr.broadcast( sci_nwes.dataset.thetao, dom_nwes.dataset.glamt.squeeze())

dep_sec = xr.DataArray( dom_nwes.get_subset_as_xarray("e3t_0",xi,yi).
                         cumsum( dim='z_dim' ).squeeze() ) 
# Extract the lat section
_, lat_sec = xr.broadcast( data_sec, dom_nwes.get_subset_as_xarray("gphit",xi,yi).squeeze() )

#%%

plt.pcolormesh(  lat_sec, dep_sec, data_sec)
plt.title('section')
plt.xlim([51,62])
#plt.clim([0, 50])
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

#%%

# Overlay contour
# Extract the coord and depth as a 1D lines
lat_lin =  dom_nwes.get_subset_as_xarray("gphit",xi,yi).squeeze() 
zd_lin = IT_obj.get_subset_as_xarray


#%%

## Make plot of pycnocline depth and stratification

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('BrBG_r')
new_cmap = truncate_colormap(cmap, 0.2, 0.8)
new_cmap.set_bad(color = 'grey')

H = IT_obj.domain.depth_t[-1,:,:].squeeze()
zd = IT_obj.zd.mean(dim='t_dim')

fig = plt.figure()
plt.rcParams['figure.figsize'] = (8.0, 8.0)


ax = fig.add_subplot(111)
cz = plt.contour( lon, lat, H, levels=[11, 80, 200, 800],
            colors=['k','k','k', 'k'],
            linewidths=[1,2,1,1])

plt.pcolormesh( lon, lat, H.where(H > 11), cmap=new_cmap) # grey land
plt.contourf( lon, lat, zd,
             levels=np.arange(0,40.+10.,10.),
             extend='max',
             cmap=new_cmap)
plt.clim([0,40.])

plt.xlim([-3,11])
plt.ylim([51,62])
plt.colorbar()


lines = [cz.collections[i] for i in range(1,len(cz.collections))] #[ cz.collections[1], cz.collections[2], cz.collections[-1] ]
labels = [str(int(cz.levels[i]))+'m' for i in range(1,len(cz.levels))]
#labels = ['80m','200m','800m']

plt.legend( lines, labels, loc="lower right")

plt.title('Pycnocline depth (m)')
plt.xlabel('longitude'); plt.ylabel('latitude')

#fig.savefig(fig_dir+'Upper200mVel.png', dpi=120)
#%%
def main():
    pass









if __name__ == "__main__": main()
