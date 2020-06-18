## scratch.py
"""
Start to try and add pycnocline diagnostics to COAsT
"""

#%%
import coast as COAsT
import numpy as np
import xarray as xa
import dask
#import matplotlib.pyplot as plt


#%%
class Diagnostics:
    def __init__(self, dataset: xa.Dataset, domain: COAsT):

        self.zt = None
        self.zd = None
        self.depth_t = domain.dataset.e3t_0.cumsum( dim='z' ).squeeze() # size: nz,my,nx
        self.depth_w = domain.dataset.e3w_0.cumsum( dim='z' ).squeeze() # size: nz,my,nx




    def difftpt2tpt(var,dim):
        """
        Compute the Euler derivative of T-pt variable onto a T-pt.
        Input the dimension index for derivative
        """
        return  0.5*( var.roll(dim=-1, roll_coords=True)
                    - var.roll(dim=+1, roll_coords=True) )


    def get_stratification(self):
        strat = difftpt2tpt( self.votemper, dim=deptht ) / difftpt2tpt( self.depth_t, dim='z' )


    def zd(var_name='votemper', var_grid='grid_T'):
        pass


    def zt():

        # load file size
        try:
            [time_size, depth_size, lat_size, lon_size] = self.dataset['votemper'].shape
        except:
            print('I assumed that votemper existed')

        # load in background stratification data
        N2_3d = fw.variables['N2_25h'][:] # (time_counter, depth, y, x). W-pts. Surface value == 0

        # Ensure surface value is 0
        N2_3d[:,0,:,:] = 0

        # Mask at level mbathy
        print(np.shape(mbathy), lat_size,lon_size)
        indexes = [[int(mbathy[JJ,II]), JJ,II] for JJ, II in [(JJ,II) for JJ in range(lat_size) for II in range(lon_size)]]
        for index in indexes:
            #print index
            N2_3d[:,index[0],index[1],index[2]] = 999
        N2_3d[np.where(N2_3d == 999)] = np.NaN


        # initialise variables
        z_d = np.zeros((time_size,lat_size,lon_size)) # pycnocline depth
        z_t = np.zeros((time_size,lat_size,lon_size)) # pycnocline thickness


        # compute pycnocline depth, thickness and dissipation at pycnocline
        # Loop over time index to make it more simple.
    #    print 'Computing pycnocline timeseries depth, thickness and dissipation'
        for time in range(time_size):
            print('time step {} of {}'.format(time, time_size))
            N2 = N2_3d[time,:,:,:]
            eps = eps_3d[time,:,:,:]



        #    if np.shape(N2) != np.shape(z):
        #        return 'inputs variables are different shapes', np.shape(N2), np.shape(z)
            if len(np.shape(N2)) != 3:
                return 'input variable does not have the expected 3 dimensions:',  np.shape(N2)


            #
            # create list of dimension sizes to tile projection
            tile_shape = [1 for i in range(len(np.shape(N2)))]
            tile_shape[0] = np.shape(N2)[ax] # replace first dimension with the size of ax dimension (number of depth levels)
                                             # [depth_size 1 ... 1]. Tile seems to work better with new dimensions at the front
            #

            intN2  = np.nansum( N2*e3w, axis=ax) # Note that N2[k=0] = 0, so ( N2*e3w )[k=0] = 0 (is good) even though e3w[k=0] inc atm
            #zw = np.cumsum( 0, e3t, axis=ax ) # Would need to add a layer of zeros on top of this cumsum
            intzN2 = np.nansum( zw*N2*e3w, axis=ax)

            z_d[time,:,:] = intzN2 / intN2 # pycnocline depth
            z_d_tile = np.tile( z_d[time,:,:], tile_shape ).swapaxes(0,ax)

            intz2N2 = np.nansum( (zw-z_d_tile)**2 * N2 * e3w, axis=ax)
        #    intz2N2 = np.trapz( (z-z_d_tile)**2 * N2, z, axis=ax)
            z_t[time,:,:] = np.sqrt(intz2N2 / intN2)
            z_t_tile = np.tile( z_t[time,:,:], tile_shape ).swapaxes(0,ax) # pycnocline thickness


            pyc_mask = (zw >= z_d_tile-z_t_tile).astype(int) * \
                       (zw <= z_d_tile+z_t_tile).astype(int)


            ndims = np.shape(N2) # store to replace max array to shape of original array
            maxarr = np.nanmax(N2*pyc_mask, axis=ax) # store to reshape final masked field with this collapsed dimension shape
            eps_pyc[time,:,:] = np.zeros(np.shape(maxarr))*np.NaN

            Nmask = (N2 == np.tile( maxarr, tile_shape ).swapaxes(0,ax) ).astype(int) # Generate boolean mask, could use *.astype(int)
            eps_pyc[time,:,:] = np.sum( np.multiply(eps,Nmask) ,axis=ax) * np.sum( np.multiply(e3w,Nmask) ,axis=ax) # Picks out epsilon at the max N depth


        return z_t

    
    def get_pyc_var(fw, zw, e3w,e3t, rho0, mbathy, ax=0):
        """

        Pycnocline depth: z_d = \int zN2 dz / \int N2 dz
        Pycnocline thickness: z_t = \sqrt{\int (z-z_d)^2 N2 dz / \int N2 dz}

        Use function to save memory

        Input:
            fw - handle for file with N2
                N2 - 3D stratification +ve [z,y,x]. W-pts. Surface value is zero
            zw - 3D depth on W-pts [z,y,x]. gdepw. Never use the top and bottom values because of masking of other variables.
            e2w
            e2t
            mbathy - used to mask bathymetry [y,x]
            ax - z dimension number

        Output:
            z_d - (t,y,x) pycnocline depth
            z_t - (t,y,x) pycnocline thickness
    #        pyc_mask - (z,y,x) 1/0 mask. Unit in pycnocline band [z_d +/- z_t]
        Useage:
            [z_d, z_t] = get_pyc_var(fw, gdepw_0, e3w,e3t, rho0, ax=0)
        """
        diag = IT( mod_var_subset, )
        return IT.zd(var_name='votemper', var_grid='grid_T')  #, IT.zt()

#%%



def strat(var, grid):
    """Compute the derivative on the provided grid"""

    pass


np.shape(sci_dom.dataset.e3t_0.cumsum(dim='z').squeeze())


def difftpt2tpt(var,dim):
    """
    Compute the Euler derivative of T-pt variable onto a T-pt.
    Input the dimension index for derivative
    """
    return  0.5*( var.roll(dim=-1, roll_coords=True) - var.roll(dim=+1, roll_coords=True) )

tt = difftpt2tpt( sci_nwes.dataset.votemper, deptht )



def difftpt2tpt(var):
    """
    Compute the Euler derivative of T-pt variable onto a T-pt.
    Input the dimension index for derivative
    """
    return  0.5*( var.roll(deptht=-1, roll_coords=True) - var.roll(deptht=+1, roll_coords=True) )

tt = difftpt2tpt( sci_nwes.dataset.votemper)




#%%

#dir = "example_files/"
dir = "/Users/jeff/downloads/"

fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'
fn_altimetry = 'COAsT_example_altimetry_data.nc'

sec = 1
subsec = 96 # Code for '`' (1 below 'a')

#%%

#################################################
## ( 1 ) Test Loading and initialising methods ##
#################################################

#-----------------------------------------------------------------------------#
# ( 1a ) Load example NEMO data (Temperature, Salinity, SSH)                  #
#                                                                             #

sci = COAsT.NEMO()
sci.load(dir + fn_nemo_dat)

sci_dom = COAsT.DOMAIN()
sci_dom.load(dir + fn_nemo_dom)

sci.set_command_variables()
sci_dom.set_command_variables()

#%%


#################################################
## Inspect data ##
#################################################
sci.dataset

#%%

#################################################
## subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
ind = sci_dom.subset_indices([50,-5], [70,10])

sci_nwes = sci.isel(y=ind[0], x=ind[1]) #nwes = northwest europe shelf
sci_nwes.set_command_variables()

dom_nwes = sci_dom.isel(y=ind[0], x=ind[1]) #nwes = northwest europe shelf
dom_nwes.set_command_variables()
#%%



def main():
    
    

    Diagnostics(sci_nwes, dom_nwes).depth_t.shape
    
if __name__ == "__main__": main()
