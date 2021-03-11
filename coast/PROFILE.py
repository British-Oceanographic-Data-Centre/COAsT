import numpy as np
import xarray as xr
from . import general_utils, plot_util, crps_util, COAsT
import matplotlib.pyplot as plt
import glob
import datetime

class PROFILE(COAsT):
    '''
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimensions:
        
        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a 
                     individual location.
        > level   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files. 
    '''
##############################################################################
###                ~ Initialisation and File Reading ~                     ###
##############################################################################

    def __init__(self):
        self.dataset = None
        return
    
    def read_EN4(self,fn_en4, multiple = False, chunks = {}):
        '''
        Reads a single or multiple EN4 netCDF files into the COAsT profile
        data structure.
        '''
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks = chunks)
        else:
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]
                
            file_to_read = []
            for file in fn_en4:
                if '*' in file:
                    wildcard_list = glob.glob(file)
                    file_to_read = file_to_read + wildcard_list
                else:
                    file_to_read.append(file)
                    
            # Reorder files to read 
            file_to_read = np.array(file_to_read)
            dates = [ff[-9:-3] for ff in file_to_read]
            dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]),1) for dd in dates]
            sort_ind = np.argsort(dates)
            file_to_read = file_to_read[sort_ind]
                    
            for ff in range(0,len(file_to_read)):
                file = file_to_read[ff]
                data_tmp = xr.open_dataset(file, chunks = chunks)
                if ff==0:
                    self.dataset = data_tmp
                else:
                    self.dataset=xr.concat((self.dataset, data_tmp),dim='N_PROF')
                
        
        rename_vars = {'LATITUDE':'latitude', 'LONGITUDE' : 'longitude',
                       'DEPH_CORRECTED' : 'depth', 'JULD':'time',
                       'POTM_CORRECTED':'potential_temperature',
                       'TEMP':'temperature', 'PSAL_CORRECTED':'practical_salinity',
                       'POTM_CORRECTED_QC':'qc_potential_temperature',
                       'PSAL_CORRECTED_QC':'qc_practical_salinity',
                       'DEPH_CORRECTED_QC':'qc_depth',
                       'JULD_QC':'qc_time'}
        rename_dims = {'N_PROF':'profile', 'N_PARAM':'parameter', 
                       'N_LEVELS':'z_dim',}
        vars_to_keep = list(rename_vars.keys())
        coords = ['LATITUDE','LONGITUDE','JULD']
        self.dataset = self.dataset.set_coords(coords)
        self.dataset = self.dataset.rename(rename_dims)
        self.dataset = self.dataset[vars_to_keep].rename_vars(rename_vars)
        return
        
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
        ind = general_utils.subset_indices_lonlat_box(self.dataset.longitude, 
                                                      self.dataset.latitude, 
                                                      lonbounds[0], lonbounds[1],
                                                      latbounds[0], latbounds[1])
        return ind
    
##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################

    def plot_profile(self, var:str, profile_indices=None ):
        
        fig = plt.figure(figsize=(7,10))
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['profile'])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(profile=ii)
            prof_depth = self.dataset.depth.isel(profile=ii)
            ax = plt.plot(prof_var, prof_depth)
            
        plt.gca().invert_yaxis()
        plt.xlabel(var + '(' + self.dataset[var].units + ')')
        plt.ylabel('Depth (' + self.dataset.depth.units + ')')
        plt.grid()
        return fig, ax
    
    def plot_map(self, profile_indices=None, var_str=None, depth_index=None):
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['profile'])
        
        profiles=self.dataset.isel(profile=profile_indices)
        
        if var_str is None:
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values, s=5)
        else:
            print(profiles)
            c = profiles[var_str].isel(level=depth_index)
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values,
                                            c = c, s=5)
        
        return fig, ax
    
    def plot_ts_diagram(self, profile_index, var_t='potential_temperature', 
                        var_s='practical_salinity'):
        
        profile = self.dataset.isel(profile=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)
        
        return fig, ax

##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################