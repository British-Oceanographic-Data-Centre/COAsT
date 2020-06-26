from .COAsT import COAsT
from dask import array
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy import interpolate



class Transect:
    
    
    def __init__(self, domain: DOMAIN, point_A: tuple, point_B: tuple,
                 dataset_T: NEMO=None, dataset_U: NEMO=None, dataset_V: NEMO=None):
        '''
        Class defining a transect between point A and point B with time dimension, 
        depth dimension and transect dimension. Data is subsetted along these dimensions.
        Point A should be of lower latitude than point B.
        
        Example usage:
            point_A = (54,-15)
            point_B = (56,-12)
            transect = coast.Transect( domain, point_A, point_B, nemo_t, nemo_u, nemo_v )
            

        Parameters
        ----------
        domain : DOMAIN, the model domain to extract transect from
        point_A : tuple, (lat,lon)
        point_B : tuple, (lat,lon)
        dataset_T : NEMO, optional, model data on the T grid
        dataset_U : NEMO, optional, model data on the U grid
        dataset_V : NEMO, optional, model data on the V grid
        
        '''
        
        # point A should be of lower latitude than point B
        if abs(point_B[0]) < abs(point_A[0]):
            self.point_A = point_B
            self.point_B = point_A
        else:
            self.point_A = point_A
            self.point_B = point_B
            
        # Get points on transect
        tran_y, tran_x = self.get_transect_indices( domain, self.point_A, self.point_B, "f" )
                
        # indices along the transect        
        self.y_idx = tran_y
        self.x_idx = tran_x
        self.len = len(tran_y)
        self.flux_across_AB = None # (time,depth_index,transect_segment_index)
        self.vol_flux_across_AB = None # (time,transect_segment_index)
        
        # dataset along the transect
        da_tran_y = xr.DataArray( tran_y, dims=['transect_dim'])
        da_tran_x = xr.DataArray( tran_x, dims=['transect_dim'])
        self.data_T = dataset_T.dataset.isel(y=da_tran_y, x=da_tran_x)
        self.data_U = dataset_U.dataset.isel(y=da_tran_y, x=da_tran_x)
        self.data_V = dataset_V.dataset.isel(y=da_tran_y, x=da_tran_x)
        # For calculations we need access to a halo of points around the transect
        # self.data_n = dataset.isel(y=tran_y+1,x=tran_x)  
        # self.data_e = dataset.isel(y=tran_y,x=tran_x+1) 
        # self.data_s = dataset.isel(y=tran_y-1,x=tran_x) 
        # self.data_w = dataset.isel(y=tran_y,x=tran_x-1) 
        
        self.domain = domain.dataset.isel(y=da_tran_y, x=da_tran_x) 
        # For calculations we need access to a halo of points around the transect
        # self.domain_n = domain.dataset.isel(y=tran_y+1,x=tran_x)  
        # self.domain_e = domain.dataset.isel(y=tran_y,x=tran_x+1) 
        # self.domain_s = domain.dataset.isel(y=tran_y-1,x=tran_x) 
        # self.domain_w = domain.dataset.isel(y=tran_y,x=tran_x-1) 
        
 
    def get_transect_indices(self, domain, point_A, point_B, grid_ref):
        '''
        Get the transect indices on a specific grid

        Parameters
        ----------
        domain : DOMAIN, the model domain to extract transect from
        point_A : tuple, (lat,lon)
        point_B : tuple, (lat,lon)
        
        Return
        ----------
        tran_y : array of y_dim indices
        tran_x : array of x_dim indices

        '''
        tran_y, tran_x, tran_len = domain.transect_indices(point_A, point_B, grid_ref="f")
        tran_y = np.asarray(tran_y)
        tran_x = np.asarray(tran_x)
        
        # Redefine transect tso that each point on the transect is seperated
        # from its neighbours by a single index change in y or x, but not both
        dist_option_1 = domain.dataset.e2f.values[0, tran_y, tran_x] + domain.dataset.e1f.values[0, tran_y+1, tran_x]
        dist_option_2 = domain.dataset.e2f.values[0, tran_y, tran_x+1] + domain.dataset.e1f.values[0, tran_y, tran_x]
        spacing = np.abs( np.diff(tran_y) ) + np.abs( np.diff(tran_x) )
        spacing[spacing!=2]=0
        doublespacing = np.nonzero( spacing )[0]
        for d_idx in doublespacing[::-1]:
            if dist_option_1[d_idx] < dist_option_2[d_idx]:
                tran_y = np.insert( tran_y, d_idx + 1, tran_y[d_idx+1] )
                tran_x = np.insert( tran_x, d_idx + 1, tran_x[d_idx] )
            else:
                tran_y = np.insert( tran_y, d_idx + 1, tran_y[d_idx] )
                tran_x = np.insert( tran_x, d_idx + 1, tran_x[d_idx+1] ) 
        return (tran_y, tran_x)
        
    
    def transport_across_AB(self):
        """
    
        Computes the flow through the transect at each segment and stores:
        transect normal velocity in   
        volume flux for each segment over the entire water column in transect.vol_flux_column_across_AB
        
        Return 
        -----------
        volume flux at each level across the entire transect
        volume flux over the entire column acros the entire transect

        """
        
        flux = np.ma.zeros(np.shape(self.data_U.vozocrtx))
        vol_flux = np.ma.zeros( np.shape(self.data_U.vozocrtx[:,0,:] ))        
        
        dy = np.diff(self.y_idx)
        dx = np.diff(self.x_idx)
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):            
            if dy[idx] > 0:
                # u flux (+ in)
                flux[:,:,idx] = self.data_U.vozocrtx[:,:,idx+1].to_masked_array() * self.domain.e2u[0,idx+1].to_masked_array()
                vol_flux[:,idx] = np.sum( flux[:,:,idx] * self.data_U.e3u[:,:,idx+1].to_masked_array(), axis=1 ) 
            elif dx[idx] > 0:
                # v flux (- in)
                flux[:,:,idx] = -self.data_V.vomecrty[:,:,idx+1].to_masked_array() * self.domain.e1v[0,idx+1].to_masked_array()
                vol_flux[:,idx] = np.sum( flux[:,:,idx] * self.data_V.e3v[:,:,idx+1].to_masked_array(), axis=1 )           
            elif dx[idx] < 0:
                # v flux (+ in)
                flux[:,:,idx] = self.data_V.vomecrty[:,:,idx].to_masked_array() * self.domain.e1v[0,idx].to_masked_array()
                vol_flux[:,idx] = np.sum( flux[:,:,idx] * self.data_V.e3v[:,:,idx].to_masked_array(), axis=1 )
        
        self.flux_across_AB = flux
        self.vol_flux_across_AB = vol_flux / 1000000.
        return ( np.sum(self.flux_across_AB, axis=2), np.sum(self.vol_flux_across_AB, axis=1) )  
    

    def moving_average(self, array_to_smooth, window=5, axis=-1):
        return convolve1d( array_to_smooth, np.ones(window), axis=axis ) / window
    
    def spatial_slice_plot(self, variable_slice, depth, smoothing_window=0, diverging_colorbar=True):  
        variable_slice = self.moving_average(variable_slice, smoothing_window, axis=-1)
        uniform_depth = np.arange(0, np.nanmax(depth), 2)
        uniform_depth_variable_slice = np.zeros( (len(uniform_depth), self.len) )
        for i in np.arange(0, self.len ):
            depth_func = interpolate.interp1d( depth[:,i], variable_slice[:,i], axis=0, bounds_error=False )        
            uniform_depth_variable_slice[:,i] = depth_func( uniform_depth )
        
        ss,zz = np.meshgrid(uniform_depth, np.arange(0,np.shape(uniform_depth_variable_slice)[1]))
        fig,ax=plt.subplots()
        if diverging_colorbar:
            lim = np.nanmax(np.abs(uniform_depth_variable_slice))
            pcol = ax.pcolormesh( zz,ss, np.transpose(uniform_depth_variable_slice),cmap='seismic',vmin=-lim,vmax=lim)
        else:
            pcol = ax.pcolormesh( zz,ss, np.transpose(uniform_depth_variable_slice),cmap='jet')
        plt.gca().invert_yaxis()
        fig.colorbar(pcol)
        plt.ylabel("Depth (m)")
        ax.set_xticks([0,variable_slice.shape[1]])
        ax.set_xticklabels(['A','B'])
    
    def transect_line_plot(self, variable, smoothing_window=0):
        fig,ax = plt.subplots()
        ax.plot(self.moving_average(variable, smoothing_window))
        ax.set_xticks([0,len(variable)])
        ax.set_xticklabels(['A','B'])
        
        
