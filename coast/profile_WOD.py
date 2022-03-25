"""Profile_WOD Class"""
from .index import Indexed
import numpy as np
import xarray as xr
from . import general_utils, plot_util
import matplotlib.pyplot as plt
import glob
import datetime
from .logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
import xarray.ufuncs as uf
import pandas as pd


class Profile_WOD(Indexed):
    """
    OBSERVATION type class for reshaping World Ocean Data (WOD) or similar that
    contains 1D profiles (profile * depth levels)  into a 2D array. 
    Note that its variable has its own dimention and in some profiles
    only some variables are present. WOD can be observed depth or a 
    standard depth as regrided by NOAA.
       Args:
        > X     --      The variable (e.g,Temperatute, Salinity, Oxygen, DIC ..)   
        > X_N    --     Dimensions of observed variable as 1D
                        (essentially number of obs variable = casts * osberved depths) 
        > casts --      Dimension for locations of observations (ie. profiles)
        > z_N   --      Dimension for depth levels of all observations as 1D
                        (essentially number of depths = casts * osberved depths) 
        > X_row_size -- Gives the vertical index (number of depths)
                        for each variable
    """

    """================Reshape to 2D================"""
    def reshape_2D(self, VAR_USER_want) :
        """reshape the 1D variable into 2D variable (profile, z_dim)
#        Args:
#            profile       ::   The profile dimention. Called cast in WOD, 
#                               common in all variables, but nans if 
                                a variable is not observed at a location
#            z_dim         ::   The dimension for depth levels.
#            VAR_USER_want ::   List of observations the user wants to reshape       
        """
         
        #find maximum z levels in any of the profiles
        D_max = int(np.max(self.dataset.z_row_size.values))
        #number of profiles
        Prof_size = self.dataset.z_row_size.shape[0]
        
        #set a 2D array (relevant to maximum depth)
        Depth_2d = np.empty((Prof_size,D_max,))
        Depth_2d[:] = np.nan
        #reshape depth information from 1D to 2D
        if np.isnan( self.dataset.z_row_size.values[0] ) == False : 
            I_SIZE = int(self.dataset.z_row_size.values[0]) 
            Depth_2d[0,:I_SIZE] = self.dataset.depth[0:I_SIZE].values
        for iJ in range( 1, Prof_size ) :
            if np.isnan( self.dataset.z_row_size.values[iJ] ) == False :
                I_START = int( np.nansum(self.dataset.z_row_size.values[:iJ]) )
                I_END = int( np.nansum(self.dataset.z_row_size.values[:iJ+1]) )
                I_SIZE = int( self.dataset.z_row_size.values[iJ] )
                Depth_2d[iJ,0:I_SIZE] = self.dataset.depth[I_START:I_END].values
        
        #check reshape
        T_OBS1 = np.delete( np.ravel(Depth_2d), np.isnan(np.ravel(Depth_2d)) ) 
        T_OBS2 = np.delete( self.dataset.depth.values, 
                            np.isnan( self.dataset.depth.values ))           
        if T_OBS1.size == T_OBS2.size and (int(np.min(T_OBS1-T_OBS2))==0 or int(np.max(T_OBS1-T_OBS2))==0) :
            print('Depth OK reshape successful')
        else:
            print('Depth WRONG!! reshape')       

                  
        #reshape obs for each variable from 1D to 2D
        VAR_ALL = np.empty((len(VAR_USER_want),Prof_size,D_max,))
        VAR_LIST = VAR_USER_want[:]
        counter_i=0
        for iN in range ( 0, len(VAR_USER_want) ) :
            print(VAR_USER_want[iN])
            #check that variable exist in the WOD observations file
            if VAR_USER_want[iN] in self.dataset :
                print("observed variable exist")
                # reshape it into 2D
                VAR_2d = np.empty((Prof_size,D_max,))
                VAR_2d[:] = np.nan
                #populate array but make sure that the indexing for number of levels
                #is not nan, as in the data there are nan indexings for number of levels
                #indicating no observations there 
                if np.isnan( self.dataset[VAR_USER_want[iN]+'_row_size'][0].values  ) == False :
                    I_SIZE = int( self.dataset[VAR_USER_want[iN]+'_row_size'][0].values ) 
                    VAR_2d[0,:I_SIZE] = self.dataset[VAR_USER_want[iN]][0:I_SIZE].values
                for iJ in range( 1, Prof_size ) :
                    if np.isnan( self.dataset[VAR_USER_want[iN]+'_row_size'].values[iJ]) == False :
                        I_START = int( np.nansum( self.dataset[VAR_USER_want[iN]+'_row_size'].values[:iJ] ) )
                        I_END = int( np.nansum( self.dataset[VAR_USER_want[iN]+'_row_size'].values[:iJ+1] ) )
                        I_SIZE = int( self.dataset[VAR_USER_want[iN]+'_row_size'].values[iJ])
                        VAR_2d[iJ,0:I_SIZE] = self.dataset[VAR_USER_want[iN]].values[I_START:I_END]
                
                #all variables in one array        
                VAR_ALL[counter_i,:,:] = VAR_2d
                counter_i=counter_i+1
                #check that you did everything correctly and the obs in yoru reshaped
                #array match the observations in original array
                T_OBS1 = np.delete( np.ravel(VAR_2d), np.isnan(np.ravel(VAR_2d)) ) 
                del I_START, I_END, I_SIZE, VAR_2d
                T_OBS2 = np.delete( self.dataset[VAR_USER_want[iN]].values, 
                                   np.isnan( self.dataset[VAR_USER_want[iN]].values ))
                           
                if T_OBS1.size == T_OBS2.size and (int(np.min(T_OBS1-T_OBS2))==0 or int(np.max(T_OBS1-T_OBS2))==0) :
                    print('OK reshape successful')
                else:
                    print('WRONG!! reshape')
                
            else:
                print("variable not in observations")
                VAR_LIST[iN]='NO'

        #REMOVE DUBLICATES 
        VAR_LIST = list(dict.fromkeys(VAR_LIST))
        #REMOVE the non-observed variables from the list of variables
        VAR_LIST.remove('NO')


        #create the new 2D dataset array
        WOD_profiles_2D = xr.Dataset(
            {
                "depth": (["profile","z_dim"], Depth_2d),
            },
            coords={
                "time" :(["profile"], self.dataset.time.values),
                "latitude" : (["profile"],self.dataset.latitude.values),
                "longitude" : (["profile"],self.dataset.longitude.values),
            },
        )
        for iN  in range ( 0, len(VAR_LIST) ) :
            WOD_profiles_2D[VAR_LIST[iN]] = ( ["profile","z_dim"],  VAR_ALL[iN,:,:] )
            
        return_prof = Profile_WOD()
        return_prof.dataset = WOD_profiles_2D
        return return_prof    



