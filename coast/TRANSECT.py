from .COAsT import COAsT
from scipy.ndimage import convolve1d
from scipy import interpolate
import gsw
import xarray as xr
import numpy as np
import math
from scipy.interpolate import griddata
from scipy.integrate import cumtrapz, trapz
from .logging_util import get_slug, debug, error
import warnings

# =============================================================================
# The TRANSECT module is a place for code related to transects only
# =============================================================================

class Transect:
    GRAVITY = 9.8 
    
    def __init__(self, point_A: tuple, point_B: tuple, nemo_F: COAsT,
                 nemo_T: COAsT=None, nemo_U: COAsT=None, nemo_V: COAsT=None):
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
        debug(f"Creating a new {get_slug(self)}")
        # point A should be of lower latitude than point B
        if abs(point_B[0]) < abs(point_A[0]):
            self.point_A = point_B
            self.point_B = point_A
        else:
            self.point_A = point_A
            self.point_B = point_B
            
        # self.nemo_F = nemo_F
        # self.nemo_U = nemo_U
        # self.nemo_V = nemo_V
        # self.nemo_T = nemo_T
        self.filename_domain = nemo_F.filename_domain
            
        # Get points on transect
        tran_y_ind, tran_x_ind = self.get_transect_indices( nemo_F )
                
        # indices along the transect        
        self.y_ind = tran_y_ind
        self.x_ind = tran_x_ind
        self.len = len(tran_y_ind)
        self.data_tran = xr.Dataset()
        
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
        debug(f"{get_slug(self)} initialised")

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
        debug(f"Fetching transect indices for {get_slug(self)} with {get_slug(nemo_F)}")
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
        debug(f"Computing transport across AB for {get_slug(self)}")  # TODO Probably want a better description here
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
                                          self.data_U.e3_0[:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_U.depth_0[:,idx+1].to_masked_array()
            elif dx[idx] > 0:
                # v flux (- in) 
                velocity[:,:,idx] = - self.data_V.vomecrty[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.data_V.e1[idx+1].to_masked_array() *
                                          self.data_V.e3_0[:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_V.depth_0[:,idx+1].to_masked_array()
            elif dx[idx] < 0:
                # v flux (+ in)
                velocity[:,:,idx] = self.data_V.vomecrty[:,:,idx].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * self.data_V.e1[idx].to_masked_array() *
                                          self.data_V.e3_0[:,idx].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                depth_0[:,idx] = self.data_V.depth_0[:,idx].to_masked_array()
        
        #dimensions = ['t_dim', 'z_dim', 'r_dim']
        
        velocity[:,:,-1] = np.nan
        depth_integrated_transport[:,-1] = np.nan
        
        self.data_tran['normal_velocities'] = xr.DataArray( velocity, 
                    coords={'time': (('t_dim'), self.data_U.time.values),'depth_0': (('z_dim','r_dim'), depth_0)},
                    dims=['t_dim', 'z_dim', 'r_dim'] )        
    
        self.data_tran['depth_integrated_transport_across_AB'] = xr.DataArray( depth_integrated_transport / 1000000.,
                    coords={'time': (('t_dim'), self.data_U.time.values)},
                    dims=['t_dim', 'r_dim'] ) 
        
        self.data_tran.depth_0.attrs['units'] = 'm'
        self.data_tran.depth_0.attrs['standard_name'] = 'Initial depth at time zero'
        self.data_tran.depth_0.attrs['long_name'] = 'Initial depth at time zero defined at the normal velocity grid points on the transect'
                        
        return  # TODO Should this return something? If not then the statement is not needed

    def __pressure_grad_fpoint(self, ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, velocity_component):
        """
        Calculates the hydrostatic and surface pressure gradients at an f-point
        along the transect, i.e. a specific value of r_dim (but for all time and depth).
        The caller must supply four datasets that define
        the hydrostatic and surface pressure at all vertical z_levels and all time 
        on the t-points around the transect i.e. for an f-point on the transect 
        defined at (j+1/2, i+1/2), we want t-points at (j,i), (j+1,i), (j,i+1), (j+1,i+1), 
        corresponding to  ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, respectively. 
        ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1 will have dimensions in time and depth.
        
        The velocity_component defines whether u or v is normal to the transect 
        for that particular segment of the transect. A segment of transect is 
        defined as being r_dim to r_dim+1 where r_dim is the dimension along the transect.


        Returns
        -------
        hpg_f : DataArray with dimensions in time and depth
            hydrostatic pressure gradient at an f-point point along the transect
            for all time and depth
        spg_f : DataArray with dimensions in time and depth
            surface pressure gradient at an f-point point along the transect

        """
        if velocity_component == "u":
            # required scale factors for derivative and averaging
            e2v = 0.5*( ds_T_j1.e2 + ds_T.e2 )
            e2v_i1 = 0.5*( ds_T_j1i1.e2 + ds_T_i1.e2 )
            e1v = 0.5*( ds_T_j1.e1 + ds_T.e1 )
            e1v_i1 = 0.5*( ds_T_j1i1.e1 + ds_T_i1.e1 )
            e1f = 0.5*( e1v + e1v_i1 )            
            # calculate gradients at v-points either side of f-point
            hpg = (ds_T_j1.pressure_h_zlevels - ds_T.pressure_h_zlevels) / e2v
            hpg_i1 = (ds_T_j1i1.pressure_h_zlevels - ds_T_i1.pressure_h_zlevels) / e2v_i1   
            # average onto f-point
            hpg_f = 0.5 * ( ( e1v * hpg ) + ( e1v_i1 * hpg_i1 ) ) / e1f 
            # as aboave            
            spg = (ds_T_j1.pressure_s - ds_T.pressure_s) / e2v
            spg_i1 = (ds_T_j1i1.pressure_s - ds_T_i1.pressure_s) / e2v_i1
            spg_f = 0.5 * ( (e1v * spg) + (e1v_i1 * spg_i1) ) / e1f 
        elif velocity_component == "v":
            # required scale factors for derivative and averaging
            e1u = 0.5 * ( ds_T_i1.e1 + ds_T.e1 ) 
            e1u_j1 = 0.5 * ( ds_T_j1i1.e1 + ds_T_j1.e1 )
            e2u = 0.5 * ( ds_T_i1.e2 + ds_T.e2 )
            e2u_j1 = 0.5 * ( ds_T_j1i1.e2 + ds_T_j1.e2 )
            e2f = 0.5 * ( e2u + e2u_j1 )
            # calculate gradients at u-points either side of f-point
            hpg = (ds_T_i1.pressure_h_zlevels - ds_T.pressure_h_zlevels) / e1u
            hpg_j1 = (ds_T_j1i1.pressure_h_zlevels - ds_T_j1.pressure_h_zlevels) / e1u_j1 
            # average onto f-point
            hpg_f = 0.5 * ( (e2u * hpg) + (e2u_j1 * hpg_j1) ) / e2f
            # as above
            spg = (ds_T_i1.pressure_s - ds_T.pressure_s) / e1u
            spg_j1 = (ds_T_j1i1.pressure_s - ds_T_j1.pressure_s) / e1u_j1 
            spg_f = 0.5 * ( (e2u * spg) + (e2u_j1 * spg_j1) ) / e2f
        
        return (hpg_f, spg_f)
    

   
    def geostrophic_transport(self, nemo_t_object: COAsT, ref_density = 1027):
        """
        This method will calculate the geostrophic velocity and volume transport
        (due to the geostrophic vurrent) across the transect. 
        4 variables are added to the TRANSECT.tran_data dataset:
            1. normal_velocity_hpg      (t_dim, depth_z_levels, r_dim)
            This is the velocity due to the hydrostatic pressure gradient
            2. normal_velocity_spg      (t_dim, r_dim)
            This is the velocity due to the surface pressure gradient
            3. transport_across_AB_hpg  (t_dim, r_dim)
            This is the volume transport due to the hydrostatic pressure gradient
            4. transport_across_AB_spg  (t_dim, r_dim
            This is the volume transport due to the surface pressure gradient
                                                                       
        Ths implementation works by regridding from s_levels to z_levels in order
        to perform the horizontal gradients. Currently the s_level depths are
        assumed fixed at their initial depths, i.e. at time zero.
        
        
        
        Parameters
        ----------
        nemo_t_object : COAsT
            This is the nemo model data on the t-grid for the entire domain. It
            must contain the temperature, salinity and t-grid domain data (e1t, e2t, e3t_0).
        ref_density : TYPE, optional
            reference density value. The default is 1027.

        Returns
        -------
        None.

        """
        debug(f"Calculating geostrophic velocity and volume transport for {get_slug(self)} with "
              f"{get_slug(nemo_t_object)}")
        nemo_T = nemo_t_object.copy()
        if 't_dim' not in nemo_T.dataset.dims:
            nemo_T_ds = nemo_T.dataset.expand_dims(dim={'t_dim':1},axis=0)
        else:
            nemo_T_ds = nemo_T.dataset 
        
        nemo_T_ds['bathymetry'] = xr.open_dataset( self.filename_domain ).bathy_metry.squeeze().rename({'y':'y_dim', 'x':'x_dim'}) 
               
        y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] ) # j
        x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] ) # i
        
        # We need to calculate the pressure at four t-points to average onto the
        # normal velocity points. Here we subset the nemo_t data around the
        # transect so we have these four t-grid points at each point along the 
        # transect
        ds_T = nemo_T_ds.isel(y_dim = y_ind, x_dim = x_ind) # j,i
        ds_T_j1 = nemo_T_ds.isel(y_dim = y_ind+1, x_dim = x_ind) # j+1,i
        ds_T_i1 = nemo_T_ds.isel(y_dim = y_ind, x_dim = x_ind+1) # j,i+1
        ds_T_j1i1 = nemo_T_ds.isel(y_dim = y_ind+1, x_dim = x_ind+1) # j+1,i+1
        
        # Construct and add the pressure fields to the 4 t-grid datasets
        self.construct_pressure(ds_T, ref_density)
        self.construct_pressure(ds_T_j1, ref_density)
        self.construct_pressure(ds_T_i1, ref_density)
        self.construct_pressure(ds_T_j1i1, ref_density)
        
        # Remove the mean hydrostatic pressure on each z_level from the hydrostatic pressure.
        # This helps to reduce the noise when taking the horizontal gradients of hydrostatic pressure.
        # Also catch and ignore nan-slice warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pressure_h_zlevel_mean = xr.concat( (ds_T.pressure_h_zlevels, ds_T_j1.pressure_h_zlevels, 
                                 ds_T_i1.pressure_h_zlevels, ds_T_j1i1.pressure_h_zlevels), 
                                 dim='concat_dim' ).mean(dim=('concat_dim','r_dim','t_dim'),skipna=True)
        ds_T['pressure_h_zlevels'] = ds_T.pressure_h_zlevels - pressure_h_zlevel_mean
        ds_T_j1['pressure_h_zlevels'] = ds_T_j1.pressure_h_zlevels - pressure_h_zlevel_mean
        ds_T_i1['pressure_h_zlevels'] = ds_T_i1.pressure_h_zlevels - pressure_h_zlevel_mean
        ds_T_j1i1['pressure_h_zlevels'] = ds_T_j1i1.pressure_h_zlevels - pressure_h_zlevel_mean
                
        # Coriolis parameter
        f = 2 * 7.2921 * 10**(-5) * np.sin( np.deg2rad(self.data_F.latitude) )
        
        dy = np.diff(self.y_ind)
        dx = np.diff(self.x_ind)
        normal_velocity_hpg = np.zeros_like(ds_T.pressure_h_zlevels)
        normal_velocity_spg = np.zeros_like(ds_T.pressure_s)
        # horizontal scale factors for each segmant of transect
        horizontal_scale = np.zeros( (ds_T.t_dim.size, ds_T.r_dim.size) ) 
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):  
            # u flux (+u is positive across transect) 
            if dy[idx] > 0:    
                # calculate the pressure gradients at two f points defining a segment of the transect                             
                hpg, spg = self.__pressure_grad_fpoint( ds_T.isel(r_dim=idx), ds_T_j1.isel(r_dim=idx),
                                        ds_T_i1.isel(r_dim=idx), ds_T_j1i1.isel(r_dim=idx), 'u' )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( ds_T.isel(r_dim=idx+1), ds_T_j1.isel(r_dim=idx+1),
                                        ds_T_i1.isel(r_dim=idx+1), ds_T_j1i1.isel(r_dim=idx+1), "u" )
                
                # average from f to u point and calculate velocities
                e2u_j1 = 0.5 * (ds_T_j1.isel(r_dim=idx).e2 + ds_T_j1i1.isel(r_dim=idx).e2 )
                u_hpg = -(0.5 * (self.data_F.e2[idx]*hpg/f[idx] + self.data_F.e2[idx+1]*hpg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))
                u_spg = -(0.5 * (self.data_F.e2[idx]*spg/f[idx] + self.data_F.e2[idx+1]*spg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))                
                normal_velocity_hpg[:,:,idx] = u_hpg.values
                normal_velocity_spg[:,idx] = u_spg.values
                horizontal_scale[:,idx] = e2u_j1
             
            # v flux (-v is positive across transect)
            elif dx[idx] > 0: 
                # calculate the pressure gradients at two f points defining a segment of the transect                                  
                hpg, spg = self.__pressure_grad_fpoint(ds_T.isel(r_dim=idx), ds_T_j1.isel(r_dim=idx),
                                        ds_T_i1.isel(r_dim=idx), ds_T_j1i1.isel(r_dim=idx), "v")
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint(ds_T.isel(r_dim=idx+1), ds_T_j1.isel(r_dim=idx+1),
                                        ds_T_i1.isel(r_dim=idx+1),ds_T_j1i1.isel(r_dim=idx+1), "v")
                
                # average from f to v point and calculate velocities
                e1v_i1 = 0.5 * ( ds_T_i1.isel(r_dim=idx).e1 + ds_T_j1i1.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_F.e1[idx]*hpg/f[idx] + self.data_F.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                v_spg = (0.5 * (self.data_F.e1[idx]*spg/f[idx] + self.data_F.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                normal_velocity_hpg[:,:,idx] = -v_hpg.values
                normal_velocity_spg[:,idx] = -v_spg.values
                horizontal_scale[:,idx] = e1v_i1
                
            # v flux (+v is positive across transect)    
            elif dx[idx] < 0:
                # calculate the pressure gradients at two f points defining a segment of the transect                                  
                hpg, spg = self.__pressure_grad_fpoint(ds_T.isel(r_dim=idx), ds_T_j1.isel(r_dim=idx),
                                        ds_T_i1.isel(r_dim=idx), ds_T_j1i1.isel(r_dim=idx), "v")
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint(ds_T.isel(r_dim=idx+1), ds_T_j1.isel(r_dim=idx+1),
                                        ds_T_i1.isel(r_dim=idx+1),ds_T_j1i1.isel(r_dim=idx+1), "v")
                
                # average from f to v point and calculate velocities
                e1v = 0.5 * ( ds_T.isel(r_dim=idx).e1 + ds_T_j1.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_F.e1[idx]*hpg/f[idx] + self.data_F.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v * ref_density))
                v_spg = (0.5 * (self.data_F.e1[idx]*spg/f[idx] + self.data_F.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v * ref_density))                   
                normal_velocity_hpg[:,:,idx] = v_hpg.values 
                normal_velocity_spg[:,idx] = v_spg.values 
                horizontal_scale[:,idx] = e1v
        
        normal_velocity_hpg[:,:,-1] = np.nan
        normal_velocity_spg[:,-1] = np.nan
        
        normal_velocity_hpg = np.where( ds_T.depth_z_levels.values[:,np.newaxis] <= ds_T.bathymetry.values, 
                               normal_velocity_hpg, np.nan )
        
        # remove redundent levels    
        active_z_levels = np.count_nonzero(~np.isnan(normal_velocity_hpg),axis=1).max() 
        normal_velocity_hpg = normal_velocity_hpg[:,:active_z_levels,:]
        z_levels = ds_T.depth_z_levels.values[:active_z_levels]
        
        coords_hpg={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), self.data_F.latitude),
                'longitude': (('r_dim'), self.data_F.longitude)}
        dims_hpg=['depth_z_levels', 'r_dim']
        attributes_hpg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the hydrostatic pressure gradient'}
        coords_spg={'latitude': (('r_dim'), self.data_F.latitude),
                'longitude': (('r_dim'), self.data_F.longitude)}
        dims_spg=['r_dim']
        attributes_spg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the surface pressure gradient'}
        
        if 't_dim' in ds_T.dims:
            coords_hpg['time'] = (('t_dim'), ds_T.time)
            dims_hpg.insert(0, 't_dim')
            coords_spg['time'] = (('t_dim'), ds_T.time)
            dims_spg.insert(0, 't_dim')
        
        self.data_tran['normal_velocity_hpg'] = xr.DataArray( np.squeeze(normal_velocity_hpg),
                coords=coords_hpg, dims=dims_hpg, attrs=attributes_hpg)
        self.data_tran['normal_velocity_spg'] = xr.DataArray( np.squeeze(normal_velocity_spg),
                coords=coords_spg, dims=dims_spg, attrs=attributes_spg)

        self.data_tran['transport_across_AB_hpg'] = ( self.data_tran
                .normal_velocity_hpg.fillna(0).integrate(dim='depth_z_levels') ) * horizontal_scale / 1000000   
        self.data_tran.transport_across_AB_hpg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the hydrostatic pressure gradient'}
        
        
        #depth_3d = self.data_tran.depth_z_levels.broadcast_like(self.data_tran.normal_velocity_hpg)
        #H = depth_3d.where(~self.data_tran.normal_velocity_hpg.to_masked_array().mask).max(dim='z_dim')
        H = ds_T.bathymetry.values
        self.data_tran['transport_across_AB_spg'] = self.data_tran.normal_velocity_spg * H * horizontal_scale / 1000000
        self.data_tran.transport_across_AB_spg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the surface pressure gradient'}
        
        return  # TODO Should this return something? If not the statement is not needed

    def construct_pressure( self, ds_T, ref_density, z_levels=None):   
        '''
            This method is for calculating the hydrostatic and surface pressure fields
            on z-levels. The motivation is to enable the calculation of horizontal 
            gradients and is primarily intended as an internal function; however, it can
            also be used with care to create pressure and density fields on a 
            transect defined on the t_grid.
            
            The ds_T argument is the t-grid dataset subsetted along a 
            transect. 
            Requirements: The supplied t-grid dataset must contain the 
            Practical Salinity and the Potential Temperature variables. The depth_0
            field must also be supplied. The GSW package is used to calculate
            The Absolute Pressure, Absolute Salinity and Conservate Temperature.
            
            Three new variables (density, hydrostatic pressure, surface pressure)
            are created and added to the ds_T dataset:
                density_zlevels       (t_dim, depth_z_levels, r_dim)
                pressure_h_zlevels    (t_dim, depth_z_levels, r_dim)
                pressure_s            (t_dim, r_dim)
            
            Note that density is constructed using the EOS10
            equation of state.

        Parameters
        ----------
        ds_T : xarray.Dataset
            the t-grid data defined on a transect.
        ref_density
            reference density value
        z_levels : (optional) numpy array
            1d array that defines the depths to interpolate the density and pressure
            on to.
        
        Returns
        -------
        None.

        '''
        # TODO Probably want a better description for this log message
        debug(f"calculating the hydrostatic and surface pressure fields on z-levels {z_levels} for {get_slug(self)}")
        if 't_dim' not in ds_T.dims:
            ds_T = ds_T.expand_dims(dim={'t_dim':1},axis=0)

        if z_levels is None:   
            bathymetry = xr.open_dataset( self.filename_domain ).bathy_metry.squeeze()
            z_levels_0_50 = np.arange(0,55,5.5)
            z_levels_60_200 = np.arange(60,210,10)
            z_levels_250_600 = np.arange(200,650,50)
            z_levels_650_ = np.arange(650,bathymetry.max()+150,150)
            z_levels = np.concatenate( (z_levels_0_50, z_levels_60_200, 
                                        z_levels_250_600, z_levels_650_) )
        
       # shape_ds = ( ds_T.t_dim.size, ds_T.z_dim.size, ds_T.r_dim.size )
        shape_ds = ( ds_T.t_dim.size, len(z_levels), ds_T.r_dim.size )
        salinity_z = np.ma.zeros( shape_ds )
        temperature_z = np.ma.zeros( shape_ds ) 
        salinity_s = ds_T.salinity.to_masked_array()
        temperature_s = ds_T.temperature.to_masked_array()
        s_levels = ds_T.depth_0.values
        
        # At the current time there does not appear to be a good algorithm for performing this 
        # type of interpolation without loops. Griddata is an option but does not
        # support extrapolation
        for it in ds_T.t_dim:
            for ir in ds_T.r_dim:
                if not np.all(np.isnan(salinity_s[it,:,ir].data)):  
                    # Need to remove the levels below the (envelope) bathymetry which are NaN
                    salinity_s_r = salinity_s[it,:,ir].compressed()
                    temperature_s_r = temperature_s[it,:,ir].compressed()
                    s_levels_r = s_levels[:len(salinity_s_r),ir]
                    
                    sal_func = interpolate.interp1d( s_levels_r, salinity_s_r, 
                                 kind='linear', fill_value="extrapolate")
                    temp_func = interpolate.interp1d( s_levels_r, temperature_s_r, 
                                 kind='linear', fill_value="extrapolate")
                    
                    salinity_z[it,:,ir] = sal_func(z_levels)
                    temperature_z[it,:,ir] = temp_func(z_levels)
                    # set levels below the bathymetry to nan
                   # salinity_z[it,:,ir] = np.where( z_levels <= ds_T.bathymetry.values[ir], 
                   #             sal_func(z_levels), np.nan )
                   # temperature_z[it,:,ir] = np.where( z_levels <= ds_T.bathymetry.values[ir], 
                   #             temp_func(z_levels), np.nan ) 
                    

        # Absolute Pressure    
        pressure_absolute = np.ma.masked_invalid(
            gsw.p_from_z( -z_levels[:,np.newaxis], ds_T.latitude ) ) # depth must be negative           
        # Absolute Salinity           
        salinity_absolute = np.ma.masked_invalid(
            gsw.SA_from_SP( salinity_z, pressure_absolute, ds_T.longitude, ds_T.latitude ) )
        salinity_absolute = np.ma.masked_less(salinity_absolute,0)
        # Conservative Temperature
        temp_conservative = np.ma.masked_invalid(
            gsw.CT_from_pt( salinity_absolute, temperature_z ) )
        # In-situ density
        density_z = np.ma.masked_invalid( gsw.rho( 
            salinity_absolute, temp_conservative, pressure_absolute ) )
        
                        
        coords={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), ds_T.latitude),
                'longitude': (('r_dim'), ds_T.longitude)}
        dims=['depth_z_levels', 'r_dim']
        attributes = {'units': 'kg / m^3', 'standard name': 'In-situ density on the z-level vertical grid'}
        
        if shape_ds[0] != 1:
            coords['time'] = (('t_dim'), ds_T.time.values)
            dims.insert(0, 't_dim')
          
        ds_T['density_zlevels'] = xr.DataArray( np.squeeze(density_z), 
                coords=coords, dims=dims, attrs=attributes )
    
        # cumulative integral of density on z levels
        # Note that zero density flux is assumed at z=0
        density_cumulative = -cumtrapz( density_z, x=-z_levels, axis=1, initial=0)

        hydrostatic_pressure = density_cumulative * self.GRAVITY
        
        attributes = {'units': 'kg m^{-1} s^{-2}', 'standard name': 'Hydrostatic pressure on the z-level vertical grid'}
        ds_T['pressure_h_zlevels'] = xr.DataArray( np.squeeze(hydrostatic_pressure), 
                coords=coords, dims=dims, attrs=attributes )
        
        ds_T['pressure_s'] = ref_density * self.GRAVITY * ds_T.ssh.squeeze()
        ds_T.pressure_s.attrs = {'units': 'kg m^{-1} s^{-2}', 
                                  'standard_name': 'surface pressure'}
        
        return

    def moving_average(self, array_to_smooth, window=2, axis=-1):  # TODO This could be a static method
        '''
        Returns the input array smoothed along the given axis using convolusion
        '''
        debug(f"Fetching moving average for {array_to_smooth}")
        return convolve1d( array_to_smooth, np.ones(window), axis=axis ) / window
    
    def interpolate_slice(self, variable_slice, depth, interpolated_depth=None ):  # TODO This could be a static method
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
        debug(f"Interpolating slice {variable_slice} at depths {depth}")
        if interpolated_depth is None:
            interpolated_depth = np.arange(0, np.nanmax(depth), 2)
            
        interpolated_depth_variable_slice = np.zeros( (len(interpolated_depth), variable_slice.shape[-1]) )
        for i in np.arange(0, variable_slice.shape[-1] ):
            depth_func = interpolate.interp1d( depth[:,i], variable_slice[:,i], axis=0, bounds_error=False )        
            interpolated_depth_variable_slice[:,i] = depth_func( interpolated_depth )
            
        return (interpolated_depth_variable_slice, interpolated_depth )  # TODO Brackets aren't required
    
    def plot_transect_on_map(self):
        '''
        Plot transect location on a map
        
        Example usage:
        --------------
        tran = coast.Transect( (54,-15), (56,-12), nemo_f, nemo_t, nemo_u, nemo_v )
        tran.plot_map()
        '''
        debug(f"Generating plot on map for {get_slug(self)}")
        try:
            import cartopy.crs as ccrs  # mapping plots
            import cartopy.feature  # add rivers, regional boundaries etc
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # deg symb
            from cartopy.feature import NaturalEarthFeature  # fine resolution coastline
        except ImportError:
            import sys
            warnings.warn("No cartopy found - please run\nconda install -c conda-forge cartopy")
            sys.exit(-1)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

        cset = plt.plot(self.data_F.longitude, self.data_F.latitude, c='k')

        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        coast = NaturalEarthFeature(category='physical', scale='50m',
                                    facecolor=[0.8,0.8,0.8], name='coastline',
                                    alpha=0.5)
        ax.add_feature(coast, edgecolor='gray')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='-')

        gl.top_labels = False
        gl.bottom_labels = True
        gl.right_labels = False
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        plt.title('Map of transect location')
        #plt.show() # Can only adjust axis if fig is plotted already..
        
        return fig, ax
    
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
        debug(f"Plotting normal velocity for {get_slug(self)} with plot_info {plot_info}")
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
        debug(f"Generating quick plot for {get_slug(self)} with plot_info {plot_info}")
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
        debug(f"Constructing in-situ density on z-levels for {get_slug(self)} with EOS \"{EOS}\"")
        try:
            if EOS != 'EOS10': 
                raise ValueError(get_slug(self) + ': Density calculation for ' + EOS + ' not implemented.')
            if self.data_T is None:
                raise ValueError(get_slug(self) + ': Density calculation can only be performed \
                    when a t-grid object has been assigned to the nemo_T attribute. This\
                    can be done at initialisation.' )
         # TODO Should this be reinstated and converted to a log message?
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
            error(err)
        return
