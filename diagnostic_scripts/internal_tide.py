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

sci = coast.NEMO(dir + fn_nemo_dat)
dom = coast.DOMAIN(dir + fn_nemo_dom)
#alt = coast.ALTIMETRY(dir + fn_altimetry)


#################################################
## subset of data and domain ##
#################################################
# Pick out a North Sea subdomain
ind = dom.subset_indices([50,-5], [70,10])

sci_nwes = sci.isel(y_dim=ind[0], x_dim=ind[1]) #nwes = northwest europe shelf
dom_nwes = dom.isel(y_dim=ind[0], x_dim=ind[1]) #nwes = northwest europe shelf



#%%

#################################################
## Create Diagnostics object
#################################################
IT_obj = coast.DIAGNOSTICS(sci_nwes, dom_nwes)

# Construct stratification
IT_obj.get_stratification( sci_nwes.dataset.votemper ) # --> self.strat

# Construct pycnocline variables: depth and thickness
IT_obj.get_pyc_vars() # --> self.zd and self.zt

#%%

#################################################
## Make Plots
#################################################
import matplotlib.pyplot as plt

plt.pcolor( IT_obj.strat[0,10,:,:]); plt.title('stratification'); plt.show()

plt.plot( IT_obj.strat[0,:,100,60],'+'); plt.title('stratification'); plt.show()

plt.plot(sci_nwes.dataset.votemper[0,:,100,60],'+'); plt.title('temperature'); plt.show()

plt.plot(IT_obj.zd[0,:,:],'+'); plt.title('pycnocline depth'); plt.show()


#IT.get_pyc_vars()


#%%

def main():
    pass









if __name__ == "__main__": main()
