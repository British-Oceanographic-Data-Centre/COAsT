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
    def __init__(self, nemo: xa.Dataset, domain: COAsT ):

        self.nemo   = nemo
        self.domain = domain
        
        # These are bespoke to the internal tide problem        
        self.zt = None
        self.zd = None
        
        # This might be generally useful and could be somewhere more accessible?
        self.strat = None
        
        # These would be generally useful and should be in the NEMO class
        self.depth_t = domain.dataset.e3t_0.cumsum( dim='zdim' ).squeeze() # size: nz,my,nx
        self.depth_w = domain.dataset.e3w_0.cumsum( dim='zdim' ).squeeze() # size: nz,my,nx


        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in zdim, ydim, xdim
        self.nz = nemo.dataset.dims['zdim']
        self.ny = nemo.dataset.dims['ydim']
        self.nx = nemo.dataset.dims['xdim']
        if domain.dataset.dims['zdim'] != self.nz:
            print('zdim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['zdim'], self.nz))
        if domain.dataset.dims['ydim'] != self.ny:
            print('ydim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['ydim'], self.ny))
        if domain.dataset.dims['xdim'] != self.nx:
            print('xdim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['xdim'], self.nx))
            
        # Create a dataset
        


    def difftpt2tpt(self, var, dim='zdim'):
        """
        Compute the Euler derivative of T-pt variable onto a T-pt.
        Input the dimension index for derivative
        """
        if dim == 'zdim':
            difference = 0.5*( var.roll(zdim=-1, roll_coords=True)
                    - var.roll(zdim=+1, roll_coords=True) )
        else:
            print('Not expecting that dimension yet')
        return difference
    
    


    def get_stratification(self, var: xa.DataArray ):
        self.strat = self.difftpt2tpt( var, dim='zdim' ) \
                    / self.difftpt2tpt( self.depth_t, dim='zdim' )
        self.zt = np.ones((self.nz, self.ny))


    def zd(var_name='votemper', var_grid='grid_T'):
        pass


    def zt( self ):

        # compute stratification 
        self.get_stratification( self.nemo.dataset.votemper )
        print('Using only temperature for stratification at the moment')
        N2_3d = self.strat  # (tdim, zdim, ydim, xdim). T-pts. Surface value == 0

        # Ensure surface value is 0
        N2_3d[:,0,:,:] = 0
        # Ensure bed value is 0
        N2_3d[:,-1,:,:] = 0        
        
        # mask out the Nan values
        N2_3d[ np.where( np.isnan(self.nemo.dataset.votemper) ) ] = np.NaN

        # initialise variables
        z_d = np.zeros((self.nt, self.ny, self.nx)) # pycnocline depth
        z_t = np.zeros((self.nt, self.ny, self.nx)) # pycnocline thickness


        # compute pycnocline depth, thickness and dissipation at pycnocline
        # Loop over time index to make it more simple.
    #    print 'Computing pycnocline timeseries depth, thickness and dissipation'
        for time in range(self.nt):
            print('time step {} of {}'.format(time, self.nt))
            N2 = N2_3d[time,:,:,:]

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

        self.zt = z_t
        self.zd = z_d

    
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
sci_nwes.set_command_dimensions()


dom_nwes = sci_dom.isel(y=ind[0], x=ind[1]) #nwes = northwest europe shelf
dom_nwes.set_command_variables()
dom_nwes.set_command_dimensions()

#%%


# Create Diagnostics object
IT = Diagnostics(sci_nwes, dom_nwes)
# Construct stratification
IT.get_stratification( sci_nwes.dataset.votemper ) # --> self.strat



import matplotlib.pyplot as plt

plt.pcolor( IT.strat[0,10,:,:]); plt.show()

plt.plot( IT.strat[0,:,100,60],'+'); plt.show()

plt.plot(sci_nwes.dataset.votemper[0,:,100,60],'+'); plt.show()
    
#%%

def main():
    pass
    




    
    
    
    
if __name__ == "__main__": main()
