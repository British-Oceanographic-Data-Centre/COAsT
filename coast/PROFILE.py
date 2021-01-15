import numpy as np
import xarray as xr
from . import general_utils, plot_util, crps_util, COAsT
import matplotlib.pyplot as plt

class PROFILE(COAsT):
    '''

    '''
##############################################################################
###                ~ Initialisation and File Reading ~                     ###
##############################################################################

    def __init__(self):
        self.dataset = None
        return
    
    def read_EN4(self,fn_en4, multiple = False, chunks = {}):
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks = chunks)
        else:
            self.dataset = xr.open_mfdataset(fn_en4, chunks = chunks)
            
        rename_vars = {'LATITUDE':'latitude', 'LONGITUDE' : 'longitude',
                       'DEPH_CORRECTED' : 'depth'}
        self.dataset = self.dataset.rename(rename_vars)
        
##############################################################################
###                ~            Manipulate           ~                     ###
##############################################################################
    
    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        lon_str = 'longitude'
        lat_str = 'latitude'
        lon = self.dataset[lon_str].values
        lat = self.dataset[lat_str].values
        ff = lon > lonbounds[0]
        ff *= lon < lonbounds[1]
        ff *= lat > latbounds[0]
        ff *= lat < latbounds[1]

        return np.where(ff)[0]
    
##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################

    def plot_profile(self, var:str, profile_indices=None ):
        
        fig = plt.figure(figsize=(7,10))
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['N_PROF'])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(N_PROF=ii)
            prof_depth = self.dataset.depth.isel(N_PROF=ii)
            ax = plt.plot(prof_var, prof_depth)
            
        plt.gca().invert_yaxis()
        plt.xlabel(var + '(' + self.dataset[var].units + ')')
        plt.ylabel('Depth (' + self.dataset.depth.units + ')')
        plt.grid()
        return fig, ax
    
    def plot_map(self, profile_indices=None):
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['N_PROF'])
        
        profiles = self.dataset[['longitude','latitude']]
        profiles=profiles.isel(N_PROF=profile_indices)
        fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                        profiles.latitude.values)
        
        return
    
    def plot_ts_diagram(self, profile_index, var_t='POTM_CORRECTED', var_s='PSAL_CORRECTED'):
        
        profile = self.dataset.isel(N_PROF=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)
        
        return fig, ax

##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################