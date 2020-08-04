from .COAsT import COAsT
from scipy.ndimage import convolve1d
from scipy import interpolate
import gsw
import xarray as xr
import numpy as np


# =============================================================================
# The TRANSECT module is a place for code related to transects only
# =============================================================================

class Transect:
    
    def __init__(self, point_A: tuple, point_B: tuple, nemo_F: COAsT,
                 nemo_T: COAsT=None, nemo_U: COAsT=None, nemo_V: COAsT=None ):
        '''
        Class defining a generic transect type, which is a 3d dataset between a point A and 
        a point B, with a time dimension, a depth dimension and a transect dimension. The 
        transect dimension defines the points along the transect.
        The model Data is subsetted in its entirety along these dimensions.
        
        Note that Point A should be of lower latitude than point B.
        
        Example usage:
            point_A = (54,-15)
            point_B = (56,-12)
            transect = coast.Transect( point_A, point_B, nemo_t, nemo_u, nemo_v, nemo_f )
            

        Parameters
        ----------
        point_A : tuple, (lat,lon)
        point_B : tuple, (lat,lon)
        nemo_F : NEMO, model data on the F grid, which is required to define the 
                    transect
        nemo_T : NEMO, optional, model data on the T grid
        nemo_U : NEMO, optional, model data on the U grid
        nemo_V : NEMO, optional, model data on the V grid
        
        
        '''
        
        # point A should be of lower latitude than point B
        if abs(point_B[0]) < abs(point_A[0]):
            self.point_A = point_B
            self.point_B = point_A
        else:
            self.point_A = point_A
            self.point_B = point_B
            
        # Get points on transect
        tran_y_ind, tran_x_ind = self.get_transect_indices( nemo_F )
                
        # indices along the transect        
        self.y_ind = tran_y_ind
        self.x_ind = tran_x_ind
        self.len = len(tran_y_ind)
        self.data_tran = xr.Dataset()
        #self.normal_velocities = None
        #self.depth_integrated_transport_across_AB = None # (time,transect_segment_index)
        
        # Subset the nemo data along the transect creating a new dimension (r_dim),
        # which is a paramterisation for x_dim and y_dim defining the transect
        da_tran_y_ind = xr.DataArray( tran_y_ind, dims=['r_dim'])
        da_tran_x_ind = xr.DataArray( tran_x_ind, dims=['r_dim'])
        self.data_F = nemo_F.dataset.isel(y_dim = da_tran_y_ind, x_dim = da_tran_x_ind)
        if nemo_T is not None:
            self.data_T = nemo_T.dataset.isel(y_dim = da_tran_y_ind, x_dim = da_tran_x_ind)
        if nemo_U is not None:
            self.data_U = nemo_U.dataset.isel(y_dim = da_tran_y_ind, x_dim = da_tran_x_ind)
        if nemo_V is not None:
            self.data_V = nemo_V.dataset.isel(y_dim = da_tran_y_ind, x_dim = da_tran_x_ind)
        # For calculations we need access to a halo of points around the transect
        # self.data_n = dataset.isel(y=tran_y+1,x=tran_x)  
        # self.data_e = dataset.isel(y=tran_y,x=tran_x+1) 
        # self.data_s = dataset.isel(y=tran_y-1,x=tran_x) 
        # self.data_w = dataset.isel(y=tran_y,x=tran_x-1) 
        
        
 
    def get_transect_indices(self, nemo_F):
        '''
        Get the transect indices on a specific grid

        Parameters
        ----------
        nemo_F : the model grid to define the transect on
        
        Return
        ----------
        tran_y : array of y_dim indices
        tran_x : array of x_dim indices

        '''
        tran_y_ind, tran_x_ind, tran_len = nemo_F.transect_indices(self.point_A, self.point_B)
        tran_y_ind = np.asarray(tran_y_ind)
        tran_x_ind = np.asarray(tran_x_ind)
        
        # Redefine transect so that each point on the transect is seperated
        # from its neighbours by a single index change in y or x, but not both
        dist_option_1 = nemo_F.dataset.e2.values[tran_y_ind, tran_x_ind] + nemo_F.dataset.e1.values[tran_y_ind+1, tran_x_ind]
        dist_option_2 = nemo_F.dataset.e2.values[tran_y_ind, tran_x_ind+1] + nemo_F.dataset.e1.values[tran_y_ind, tran_x_ind]
        spacing = np.abs( np.diff(tran_y_ind) ) + np.abs( np.diff(tran_x_ind) )
        spacing[spacing!=2]=0
        doublespacing = np.nonzero( spacing )[0]
        for ispacing in doublespacing[::-1]:
            if dist_option_1[ispacing] < dist_option_2[ispacing]:
                tran_y_ind = np.insert( tran_y_ind, ispacing+1, tran_y_ind[ispacing+1] )
                tran_x_ind = np.insert( tran_x_ind, ispacing+1, tran_x_ind[ispacing] )
            else:
                tran_y_ind = np.insert( tran_y_ind, ispacing+1, tran_y_ind[ispacing] )
                tran_x_ind = np.insert( tran_x_ind, ispacing+1, tran_x_ind[ispacing+1] ) 
        return (tran_y_ind, tran_x_ind)
        
    
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
        depth_0 = np.ma.zeros(np.shape(self.data_U.depth_0))
        
        dy = np.diff(self.y_ind)
        dx = np.diff(self.x_ind)
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):            
            if dy[idx] > 0:
                # u flux (+ in)
                velocity[:,:,idx] = self.data_U.vozocrtx[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.data_U.e2[idx+1].to_masked_array() *
                                          self.data_U.e3[:,:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_U.depth_0[:,idx+1].to_masked_array()
            elif dx[idx] > 0:
                # v flux (- in) 
                velocity[:,:,idx] = - self.data_V.vomecrty[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.data_V.e1[idx+1].to_masked_array() *
                                          self.data_V.e3[:,:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_V.depth_0[:,idx+1].to_masked_array()
            elif dx[idx] < 0:
                # v flux (+ in)
                velocity[:,:,idx] = self.data_V.vomecrty[:,:,idx].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.data_V.e1[idx].to_masked_array() *
                                          self.data_V.e3[:,:,idx].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_V.depth_0[:,idx].to_masked_array()
        
        
        dimensions = ['t_dim', 'z_dim', 'r_dim']
        
  #      self.data_tran['depth_0'] = xr.DataArray( depth_0[:,:-1], dims=['z_dim', 'r_dim'], 
   #                attrs={'Units':'m', 'standard_name': 'Initial depth at time zero',
    #                'long_name': 'Initial depth at time zero defined at the normal velocity grid points on the transect'})
 
   
        self.data_tran['normal_velocities'] = xr.DataArray( velocity[:,:,:-1], 
                    coords={'time': (('t_dim'), self.data_U.time.values),'depth_0': (('z_dim','r_dim'), depth_0[:,:-1])},
                    dims=['t_dim', 'z_dim', 'r_dim'] )        
    
        self.data_tran['depth_integrated_transport_across_AB'] = xr.DataArray( depth_integrated_transport[:,:-1] / 1000000.,
                    coords={'time': (('t_dim'), self.data_U.time.values)},
                    dims=['t_dim', 'r_dim'] ) 
        
        self.data_tran.depth_0.attrs['units'] = 'm'
        self.data_tran.depth_0.attrs['standard_name'] = 'Initial depth at time zero'
        self.data_tran.depth_0.attrs['long_name'] = 'Initial depth at time zero defined at the normal velocity grid points on the transect'
                        
        return  
    

    def moving_average(self, array_to_smooth, window=2, axis=-1):
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
            
        interpolated_depth_variable_slice = np.zeros( (len(interpolated_depth), variable_slice.shape[-1]) )
        for i in np.arange(0, variable_slice.shape[-1] ):
            depth_func = interpolate.interp1d( depth[:,i], variable_slice[:,i], axis=0, bounds_error=False )        
            interpolated_depth_variable_slice[:,i] = depth_func( interpolated_depth )
            
        return (interpolated_depth_variable_slice, interpolated_depth )
    
    
    def plot_normal_velocity(self, time, plot_info: dict, cmap, smoothing_window=0):  
        '''
            Quick plot routine of velocity across the transect AB at a specific time.
            An option is provided to smooth the velocities along the transect.
            NOTE: For smoothing use even integers to smooth the x and y velocities together
    
    
    
        Parameters
        ---------------
        time: either as integer index or actual time as a string.
        plot_info: dictionary of infomation {'fig_size': value, 'title': value, 'vmin':value, 'vmax':value}
        Note that if vmin and max are not set then the colourbar will be centred at zero
        smoothing_window: smoothing via convolusion, larger number applies greater smoothing, recommended
        to use even integers

        
        '''
        try:
            data = self.data_tran.sel(t_dim = time)
        except KeyError:
            data = self.data_tran.isel(t_dim = time)        
        
        if smoothing_window != 0:
            normal_velocities, depth = self.interpolate_slice( data.normal_velocities, data.depth_0 )            
            normal_velocities = self.moving_average(normal_velocities, smoothing_window, axis=-1)
            r_dim_2d = np.broadcast_to( data.r_dim, normal_velocities.shape  )
        else:
            normal_velocities = data.normal_velocities
            depth = data.depth_0
            _ , r_dim_2d = xr.broadcast( depth, data.r_dim  )
                    
        import matplotlib.pyplot as plt
        plt.close('all')
        fig = plt.figure(figsize=plot_info['fig_size'])
        ax = fig.gca()

        plt.pcolormesh(r_dim_2d, depth, normal_velocities, cmap=cmap)
            
        plt.title(plot_info['title'])
        plt.ylabel('Depth [m]')
        try:
            plt.clim(vmin=plot_info['vmin'], vmax=plot_info['vmax'])
        except KeyError:
            lim = np.nanmax(np.abs(normal_velocities))
            plt.clim(vmin=-lim, vmax=lim)
        plt.xticks([0,data.r_dim.values[-1]],['A','B'])
        plt.colorbar(label='Velocities across AB [m/s]')
        plt.gca().invert_yaxis()

        plt.show()
        return fig,ax
        
    
    def plot_depth_integrated_transport(self, time, plot_info: dict, smoothing_window=0):
        '''
            Quick plot routine of depth integrated transport across the transect AB at a specific time.
            An option is provided to smooth along the transect via convolution, 
            NOTE: For smoothing use even integers to smooth the x and y velocities together
    
        Parameters
        ---------------
        time: either as integer index or actual time as a string.
        plot_info: dictionary of infomation {'fig_size': value, 'title': value}
        smoothing_window: smoothing via convolusion, larger number applies greater smoothing. 
        Recommended to use even integers.
        returns: pyplot object
        '''
        try:
            data = self.data_tran.sel(t_dim = time)
        except KeyError:
            data = self.data_tran.isel(t_dim = time)            
        
        if smoothing_window != 0:    
            transport = self.moving_average(data.depth_integrated_transport_across_AB, smoothing_window, axis=-1)
        else:
            transport = data.depth_integrated_transport_across_AB
        
        import matplotlib.pyplot as plt
        plt.close('all')
        fig = plt.figure(figsize=plot_info['fig_size'])
        ax = fig.gca()

        plt.plot( data.r_dim, transport )

        plt.title(plot_info['title'])
        plt.xticks([0,data.r_dim[-1]],['A','B'])
        plt.ylabel('Volume transport across AB [SV]')
        plt.show()
        return fig,ax
    
    
    def construct_density_on_z_levels( self, EOS='EOS10'):#, z_levels=None ):        
        '''
            For s-level model output this method recontructs the in-situ density 
            onto z_levels along the transect. The z_levels and density field
            are added to the data_T dataset attribute. 
            
            Requirements: The supplied t-grid dataset must contain the 
            Practical Salinity and the Potential Temperature variables. The depth_0
            field must also be supplied. The GSW package is used to calculate
            The Absolute Pressure, Absolute Salinity and Conservate Temperature.
            
            This method is useful when horizontal gradients of the density field 
            are required. Currently z_levels cannot be specified, the 
            method will contruct a z_levels profile from the s_levels.
            
            Note that currently density can only be constructed using the EOS10
            equation of state.

        Parameters
        ----------
        EOS : equation of state, optional
            DESCRIPTION. The default is 'EOS10'.


        Returns
        -------
        None.
        adds attributes Transect.data_T.depth_z_levels and Transect.data_T.density_z_levels

        '''        

        
        try:
            if EOS != 'EOS10': 
                raise ValueError(str(self) + ': Density calculation for ' + EOS + ' not implemented.')
            if self.data_T is None:
                raise ValueError(str(self) + ': Density calculation can only be performed for a t-grid object,\
                                 the tracer grid for NEMO.' )
          #  if not self.data_T.ln_sco.item():
           #     raise ValueError(str(self) + ': Density calculation only implemented for s-vertical-coordinates.')            
    
            #if z_levels is None:
            z_levels = self.data_T.depth_0.max(dim=(['r_dim']))                
            z_levels_min = self.data_T.depth_0[0,:].max(dim=(['r_dim']))
            z_levels[0] = z_levels_min
            #else:
             #   z_max = self.data_T.depth_0.max(dim=(['r_dim','z_dim'])).item()
              #  z_min = self.data_T.depth_0[0,:].max(dim=(['r_dim'])).item()
               # z_levels = z_levels[z_levels<=z_max]
                #z_levels = z_levels[z_levels>=z_min]
                
                
            
            try:    
                shape_ds = ( self.data_T.t_dim.size, z_levels.size, 
                                self.data_T.r_dim.size )
                sal = self.data_T.salinity.to_masked_array()
                temp = self.data_T.temperature.to_masked_array()                
            except AttributeError:
                shape_ds = ( 1, z_levels.size, self.data_T.r_dim.size )
                sal = self.data_T.salinity.to_masked_array()[np.newaxis,...]
                temp = self.data_T.temperature.to_masked_array()[np.newaxis,...]
            
            sal_z_levels = np.ma.zeros( shape_ds )
            temp_z_levels = np.ma.zeros( shape_ds )
            density_z_levels = np.ma.zeros( shape_ds )
            
            s_levels = self.data_T.depth_0.to_masked_array()
            lat = self.data_T.latitude.values
            lon = self.data_T.longitude.values
    
            for it in np.arange(0, shape_ds[0]):
                for ir in self.data_T.r_dim:
                    if np.all(np.isnan(sal[it,:,ir])):
                        density_z_levels[it,:,ir] = np.nan
                        density_z_levels[it,:,ir].mask = True
                    else:                      
                        sal_func = interpolate.interp1d( s_levels[:,ir], sal[it,:,ir], 
                                    bounds_error=False, kind='linear')
                        temp_func = interpolate.interp1d( s_levels[:,ir], temp[it,:,ir], 
                                    bounds_error=False, kind='linear')
                        
                        sal_z_levels[it,:,ir] = sal_func(z_levels.values)
                        temp_z_levels[it,:,ir] = temp_func(z_levels.values)
                        
            # Absolute Pressure    
            pressure_absolute = np.ma.masked_invalid(
                gsw.p_from_z( -z_levels.values[:,np.newaxis], lat ) ) # depth must be negative           
            # Absolute Salinity           
            sal_absolute = np.ma.masked_invalid(
                gsw.SA_from_SP( sal_z_levels, pressure_absolute, lon, lat ) )
            sal_absolute = np.ma.masked_less(sal_absolute,0)
            # Conservative Temperature
            temp_conservative = np.ma.masked_invalid(
                gsw.CT_from_pt( sal_absolute, temp_z_levels ) )
            # In-situ density
            density_z_levels = np.ma.masked_invalid( gsw.rho( 
                sal_absolute, temp_conservative, pressure_absolute ) )
            
            coords={'depth_z_levels': (('z_dim'), z_levels.values),
                    'latitude': (('r_dim'), self.data_T.latitude.values),
                    'longitude': (('r_dim'), self.data_T.longitude.values)}
            dims=['z_dim', 'r_dim']
            attributes = {'units': 'kg / m^3', 'standard name': 'In-situ density on the z-level vertical grid'}
            
            if shape_ds[0] != 1:
                coords['time'] = (('t_dim'), self.data_T.time.values)
                dims.insert(0, 't_dim')
              
            self.data_T['density_z_levels'] = xr.DataArray( np.squeeze(density_z_levels), 
                    coords=coords, dims=dims, attrs=attributes )

        except AttributeError as err:
            print(err)
            
        return
        
        
