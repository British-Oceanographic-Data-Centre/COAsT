## internal_tide.py
"""
Script to demonstrate internal tide diagnostics using the COAsT package.

This is a work in progress, more to demonstrate a concept than an exemplar of
 good coding or even the package's functionality.

This would form the template for HTML tutorials.
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

#################################################
## Loading and initialising methods ##
#################################################

#sci = coast.NEMO(dir + fn_nemo_dat)
#dom = coast.DOMAIN(dir + fn_nemo_dom)
##alt = coast.ALTIMETRY(dir + fn_altimetry)

#sci = coast.NEMO("/projectsa/COAsT/ind_1h_20180101_20160109_shelftmb_grid_T.nc")
#dom = coast.DOMAIN("/projectsa/accord/BoBEAS/INPUTS/domain_cfg.nc")

#sci = coast.NEMO("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_20100130_20100203_grid_T.nc")
sci = coast.NEMO("/projectsa/NEMO/jelt/AMM60_ARCHER_DUMP/AMM60smago/EXP_NSea/OUTPUT/AMM60_1d_20100704_20100708_grid_T.nc") 
dom = coast.DOMAIN("/projectsa/FASTNEt/jelt/AMM60/mesh_mask.nc")

#################################################
## subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
ind = dom.subset_indices([51,-4], [62,10])

#ind = dom.subset_indices([10,90], [15,95]) # Andermann Shelf

dom_nwes = dom.isel(y_dim=ind[0], x_dim=ind[1]) #nwes = northwest europe shelf


#%%
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

ind_sci = sci.subset_indices([51,-4], [62,10])

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

#%%

[JJ,II] = sci_nwes.find_j_i( lat= 55, lon= 5, grid_ref='t')

# short cuts for variable names
lon =  IT_obj.domain.dataset.glamt.squeeze()
lat =  IT_obj.domain.dataset.gphit.squeeze()
dep =  IT_obj.depth_t[:,:,:]

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
def main():
    pass









if __name__ == "__main__": main()
