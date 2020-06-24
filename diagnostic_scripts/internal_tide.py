## internal_tide.py
"""
Script to demonstrate internal tide diagnostics using the COAsT package.

This is a work in progress, more to demonstrate a concept than an exemplar of
 good coding or even the package's functionality.

This would form the template for HTML tutorials. 
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
        self.depth_t = domain.dataset.e3t_0.cumsum( dim='z_dim' ).squeeze() # size: nz,my,nx
        self.depth_w = domain.dataset.e3w_0.cumsum( dim='z_dim' ).squeeze() # size: nz,my,nx


        # Define the spatial dimensional size and check the dataset and domain arrays are the same size in z_dim, ydim, xdim
        self.nt = nemo.dataset.dims['t_dim']
        self.nz = nemo.dataset.dims['z_dim']
        self.ny = nemo.dataset.dims['y_dim']
        self.nx = nemo.dataset.dims['x_dim']
        if domain.dataset.dims['z_dim'] != self.nz:
            print('z_dim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['z_dim'], self.nz))
        if domain.dataset.dims['y_dim'] != self.ny:
            print('ydim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['y_dim'], self.ny))
        if domain.dataset.dims['x_dim'] != self.nx:
            print('xdim domain data size (%s) differs from nemo data size (%s)'
                  %(domain.dataset.dims['x_dim'], self.nx))
            
        # Create a dataset
        


    def difftpt2tpt(self, var, dim='z_dim'):
        """
        Compute the Euler derivative of T-pt variable onto a T-pt.
        Input the dimension index for derivative
        """
        if dim == 'z_dim':
            difference = 0.5*( var.roll(z_dim=-1, roll_coords=True)
                    - var.roll(z_dim=+1, roll_coords=True) )
        else:
            print('Not expecting that dimension yet')
        return difference
    
    


    def get_stratification(self, var: xa.DataArray ):
        self.strat = self.difftpt2tpt( var, dim='z_dim' ) \
                    / self.difftpt2tpt( self.depth_t, dim='z_dim' )


    def get_pyc_vars(self):
        """

        Pycnocline depth: z_d = \int zN2 dz / \int N2 dz
        Pycnocline thickness: z_t = \sqrt{\int (z-z_d)^2 N2 dz / \int N2 dz}


        Input:
            fw - handle for file with N2
                N2 - 3D stratification +ve [z,y,x]. W-pts. Surface value is zero
            zw - 3D depth on W-pts [z,y,x]. gdepw. Never use the top and bottom values because of masking of other variables.
            e2w
            e2t
            mbathy - used to mask bathymetry [y,x]
            ax - z dimension number

        Output:
            self.z_d - (t,y,x) pycnocline depth
            self.z_t - (t,y,x) pycnocline thickness
        Useage:
            ...
        """

        # compute stratification 
        self.get_stratification( self.nemo.dataset.votemper )
        print('Using only temperature for stratification at the moment')
        N2_4d = self.strat  # (t_dim, z_dim, ydim, xdim). T-pts. Surface value == 0

        # Ensure surface value is 0
        N2_4d[:,0,:,:] = 0
        # Ensure bed value is 0
        N2_4d[:,-1,:,:] = 0        
        
        # mask out the Nan values
        N2_4d = N2_4d.where( xa.ufuncs.isnan(self.nemo.dataset.votemper), drop=True )
        #N2_4d[ np.where( np.isnan(self.nemo.dataset.votemper) ) ] = np.NaN

        # initialise variables
        z_d = np.zeros((self.nt, self.ny, self.nx)) # pycnocline depth
        z_t = np.zeros((self.nt, self.ny, self.nx)) # pycnocline thickness

        
        # Broadcast to fill out missing (time) dimensions in grid data
        _, depth_t_4d = xa.broadcast(N2_4d, self.depth_t)
        _, depth_w_4d = xa.broadcast(N2_4d, self.depth_w)
        _, e3t_0_4d = xa.broadcast(N2_4d, self.domain.dataset.e3t_0.squeeze())

        
        
        # intergrate strat over depth
        intN2  = ( N2_4d * e3t_0_4d ).sum( dim='z_dim', skipna=True)
        # intergrate (depth * strat) over depth
        intzN2 = (N2_4d * e3t_0_4d * depth_t_4d).sum( dim='z_dim', skipna=True)
        

        # compute pycnocline depth
        z_d = intzN2 / intN2 # pycnocline depth
        
        # compute pycnocline thickness
        intz2N2 = ( xa.ufuncs.square(depth_t_4d - z_d) * e3t_0_4d * N2_4d  ).sum( dim='z_dim', skipna=True )
        #    intz2N2 = np.trapz( (z-z_d_tile)**2 * N2, z, axis=ax)
        z_t = xa.ufuncs.sqrt(intz2N2 / intN2)

        
        self.zt = z_t
        self.zd = z_d


#%%









#%%

#dir = "example_files/"
dir = "/Users/jeff/downloads/"

fn_nemo_dat = 'COAsT_example_NEMO_data.nc'
fn_nemo_dom = 'COAsT_example_NEMO_domain.nc'



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

IT.get_pyc_vars

import matplotlib.pyplot as plt

plt.pcolor( IT.strat[0,10,:,:]); plt.show()

plt.plot( IT.strat[0,:,100,60],'+'); plt.show()

plt.plot(sci_nwes.dataset.votemper[0,:,100,60],'+'); plt.show()

IT.get_pyc_vars()
    
#%%

def main():
    pass
    




    
    
    
    
if __name__ == "__main__": main()
