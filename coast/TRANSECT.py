from .COAsT import COAsT
from scipy.ndimage import convolve1d
from scipy import interpolate
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# The TRANSECT module is a place for code related to transects only
# =============================================================================

class Transect:
    
    def __init__(self, domain: COAsT, point_A: tuple, point_B: tuple,
                 dataset_T: COAsT=None, dataset_U: COAsT=None, dataset_V: COAsT=None):
        '''
        Class defining a generic transect type, which is a 3d dataset between a point A and 
        a point B, with a time dimension, a depth dimension and a transect dimension. The 
        transect dimension defines the points along the transect.
        The model Data is subsetted in its entirety along these dimensions.
        
        Note that Point A should be of lower latitude than point B.
        
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
        self.normal_velocities = None
        self.depth_integrated_transport_across_AB = None # (time,transect_segment_index)
        
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
        
        # Redefine transect so that each point on the transect is seperated
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
        Transect normal velocities at each grid point in Transect.normal_velocities,
        Depth integrated volume transport across the transect at each transect segment in 
        Transect.depth_integrated_transport_across_AB
        
        Return 
        -----------
        Transect normal velocities at each grid point (m/s)
        Depth integrated volume transport across the transect at each transect segment (Sv)
        """
        
        velocity = np.ma.zeros(np.shape(self.data_U.vozocrtx))
        vol_transport = np.ma.zeros(np.shape(self.data_U.vozocrtx))
        depth_integrated_transport = np.ma.zeros( np.shape(self.data_U.vozocrtx[:,0,:] ))        
        
        dy = np.diff(self.y_idx)
        dx = np.diff(self.x_idx)
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):            
            if dy[idx] > 0:
                # u flux (+ in)
                velocity[:,:,idx] = self.data_U.vozocrtx[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.domain.e2u[0,idx+1].to_masked_array() *
                                          self.data_U.e3u[:,:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
            elif dx[idx] > 0:
                # v flux (- in) 
                velocity[:,:,idx] = - self.data_V.vomecrty[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.domain.e1v[0,idx+1].to_masked_array() *
                                          self.data_V.e3v[:,:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
            elif dx[idx] < 0:
                # v flux (+ in)
                velocity[:,:,idx] = self.data_V.vomecrty[:,:,idx].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.domain.e1v[0,idx].to_masked_array() *
                                          self.data_V.e3v[:,:,idx].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
        
        self.normal_velocities = velocity
        self.depth_integrated_transport_across_AB = depth_integrated_transport / 1000000.
        return ( self.normal_velocities, self.depth_integrated_transport_across_AB )  
    

    def moving_average(self, array_to_smooth, window=5, axis=-1):
        '''
        Returns the input array smoothed along the given axis using convolusion
        '''
        return convolve1d( array_to_smooth, np.ones(window), axis=axis ) / window
    
    def interpolate_slice(self, variable_slice, depth, interpolated_depth=None ):
        '''
        Linearly interpolates the variable at a single time along the z_dim, which must be the
        first axis.

        Parameters
        ----------
        variable_slice : Variable to interpolate (z_dim, transect_dim)
        depth : The depth at each z point for each point along the transect
        interpolated_depth : (optional) desired depth profile to interpolate to. If not supplied
            a uniform depth profile uniformaly spaced between zero and variable max depth will be used
            with a spacing of 2 metres.

        Returns
        -------
        interpolated_depth_variable_slice : Interpolated variable
        interpolated_depth : Interpolation depth

        '''
        if interpolated_depth is None:
            interpolated_depth = np.arange(0, np.nanmax(depth), 2)
            
        interpolated_depth_variable_slice = np.zeros( (len(interpolated_depth), self.len) )
        for i in np.arange(0, self.len ):
            depth_func = interpolate.interp1d( depth[:,i], variable_slice[:,i], axis=0, bounds_error=False )        
            interpolated_depth_variable_slice[:,i] = depth_func( interpolated_depth )
            
        return (interpolated_depth_variable_slice, interpolated_depth )
    
    def spatial_slice_plot(self, variable_slice, depth, smoothing_window=0, diverging_colorbar=True, colorbarlabel:str=None,
                           title:str=None):  
        '''
            Quick plot routine of a variable along the transect. The variable is regridded onto uniformaly spaced
            vertical coordinates. An option is provided to smooth the regridded variables along the transect. The
            diverging colorbar option will centre the colorbar at zero.
    
        Parameters
        ---------------
        variable_slice: variable at a single time with dimensions (z_dim, transect_dim)
        depth: the depth field at each point
        smoothing_window: smoothing via convolusion, larger number applies greater smoothing
        diverging_colorbar: True sets the colorbar limits to centre zero with a diverging colormap
        
        '''
        
        interpolated_depth_variable_slice, interpolated_depth = self.interpolate_slice(variable_slice, depth)
        interpolated_depth_variable_slice = self.moving_average(interpolated_depth_variable_slice, smoothing_window, axis=-1)
        ss,zz = np.meshgrid(interpolated_depth, np.arange(0,np.shape(interpolated_depth_variable_slice)[1]))
        fig,ax=plt.subplots()
        if diverging_colorbar:
            lim = np.nanmax(np.abs(interpolated_depth_variable_slice))
            pcol = ax.pcolormesh( zz,ss, np.transpose(interpolated_depth_variable_slice),cmap='seismic',vmin=-lim,vmax=lim)
        else:
            pcol = ax.pcolormesh( zz,ss, np.transpose(interpolated_depth_variable_slice),cmap='jet')
        
        CB = plt.colorbar(pcol)
        CB.ax.set_ylabel(colorbarlabel)
        ax.set_ylabel("Depth (m)")
        ax.set_title(title)
        ax.set_xticks([0,variable_slice.shape[1]])
        ax.set_xticklabels(['A','B'])        
        plt.gca().invert_yaxis()
        
    
    def transect_line_plot(self, variable, smoothing_window=0, y_label:str=None,
                           title:str=None):
        '''
        A simple line plot of the variable at a single time along the transect. Smoothing can be applied
        via convolusion, where a larger number applies greater smoothing.

        '''
        fig,ax = plt.subplots()
        ax.plot(self.moving_average(variable, smoothing_window))
        ax.set_xticks([0,len(variable)])
        ax.set_xticklabels(['A','B'])
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        
