from .COAsT import COAsT
from scipy.ndimage import convolve1d
from scipy import interpolate
import gsw
import xarray as xr
import xarray.ufuncs as uf
import numpy as np
import math
from scipy.interpolate import griddata
from scipy.integrate import cumtrapz, trapz
import warnings
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt 
from skimage import measure


# =============================================================================
# The contour module is a place for code related to contours only
# =============================================================================

class Contour:
    GRAVITY = 9.8 
    
    @staticmethod
    def get_contours(nemo: COAsT, contour_depth):
        contours = measure.find_contours( nemo.dataset.bathymetry, contour_depth )
        # The find_contours method returns indices that have been interpolated
        # between grid points so we must round and cast to integer 
        contours = [np.round(contour).astype(int) for contour in contours]
        return contours, len(contours)
    
    @staticmethod
    def plot_contour(nemo: COAsT, contour):
        fig,ax=plt.subplots()
        lat = nemo.dataset.latitude[xr.DataArray(contour[:,0]), xr.DataArray(contour[:,1])]
        lon = nemo.dataset.longitude[xr.DataArray(contour[:,0]), xr.DataArray(contour[:,1])]
        
        nemo.dataset.bathymetry.where(nemo.dataset.bathymetry > 0, np.nan) \
            .plot.pcolormesh(y='latitude',x='longitude',ax=ax)
        ax.scatter(lon,lat, s=0.5, color='r')
        
    @staticmethod    
    def refine_contour(nemo: COAsT, contour, start_coords, end_coords):
        y_ind = contour[:,0]
        x_ind = contour[:,1]
        # Create tree of lat and lon on the pre-processed contour
        bt = BallTree( np.deg2rad( list( zip( nemo.dataset.latitude.values[y_ind, x_ind], 
                    nemo.dataset.longitude.values[y_ind, x_ind] ) ) ), metric='haversine' )

        # Get start and end indices for contour and subset accordingly
        start_idx = bt.query( np.deg2rad( [start_coords]) )[1][0][0]
        end_idx = bt.query( np.deg2rad([end_coords]) )[1][0][0]   
        y_ind = y_ind[start_idx:end_idx+1] 
        x_ind = x_ind[start_idx:end_idx+1]
        return y_ind, x_ind, np.vstack((y_ind,x_ind)).T
    
    def __init__(self, nemo: COAsT, y_ind, x_ind, depth=None):        
        '''
        

        Parameters
        ----------
        nemo : COAsT
            DESCRIPTION.
        y_ind : TYPE
            DESCRIPTION.
        x_ind : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.y_ind, self.x_ind = self.process_contour( nemo.dataset, y_ind, x_ind )
        self.len = len(self.y_ind)
        self.depth = depth
        self.filename_domain = nemo.filename_domain
        da_y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] )
        da_x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] )
        self.data_contour = nemo.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        

  
    
    def process_contour(self, ds, y_ind, x_ind):
        # Redefine transect so that each point on the transect is seperated
        # from its neighbours by a single index change in y or x, but not both
        dist_option_1 = ds.e2[xr.DataArray(y_ind), xr.DataArray(x_ind)] + ds.e1[xr.DataArray(y_ind), xr.DataArray(x_ind)]
        dist_option_2 = ds.e2[xr.DataArray(y_ind), xr.DataArray(x_ind+1)] + ds.e1[xr.DataArray(y_ind), xr.DataArray(x_ind)]
        spacing = np.abs( np.diff(y_ind) ) + np.abs( np.diff(x_ind) )
        spacing[spacing!=2]=0
        doublespacing = np.nonzero( spacing )[0]
        for ispacing in doublespacing[::-1]:
            if dist_option_1[ispacing] < dist_option_2[ispacing]:
                y_ind = np.insert( y_ind, ispacing+1, y_ind[ispacing+1] )
                x_ind = np.insert( x_ind, ispacing+1, x_ind[ispacing] )
            else:
                y_ind = np.insert( y_ind, ispacing+1, y_ind[ispacing] )
                x_ind = np.insert( x_ind, ispacing+1, x_ind[ispacing+1] ) 
        
        # Remove any repeated points caused by the rounding of the indices
        nonrepeated_idx = np.nonzero( np.abs( np.diff(y_ind) ) + np.abs( np.diff(x_ind) ) )

        y_ind = np.concatenate( (y_ind[nonrepeated_idx], [y_ind[-1]]) )
        x_ind = np.concatenate( (x_ind[nonrepeated_idx], [x_ind[-1]]) )
        
        print(len(y_ind))
        print(type(y_ind))
        if y_ind[0] > y_ind[-1]:
            y_ind = y_ind[::-1]
            x_ind = x_ind[::-1]

        return (y_ind, x_ind)
    
    
    def gen_z_levels(self, max_depth):
        max_depth = max_depth + 650
        z_levels_0_50 = np.arange(0,55,5)
        z_levels_60_290 = np.arange(60,300,10)
        z_levels_300_600 = np.arange(300,650,50)
        z_levels_650_ = np.arange(650,max_depth+150,150)
        z_levels = np.concatenate( (z_levels_0_50, z_levels_60_290, 
                                    z_levels_300_600, z_levels_650_) )
        z_levels = z_levels[z_levels <= max_depth] 
        return z_levels
    

class Contour_f(Contour):
    def __init__(self, nemo_f: COAsT, y_ind, x_ind, depth=None):
        super().__init__(nemo_f, y_ind, x_ind, depth)
        self.data_cross_flow = xr.Dataset()
        
    def calc_cross_contour_flow(self, nemo_u: COAsT, nemo_v: COAsT):
        """
    
        Computes the flow across the contour at each segment and stores:
        Transect normal velocities at each grid point in Transect.normal_velocities,
        Depth integrated volume transport across the transect at each transect segment in 
        Transect.depth_integrated_transport_across_AB
        
        Return 
        -----------
        Transect normal velocities at each grid point (m/s)
        Depth integrated volume transport across the transect at each transect segment (Sv)
        """
        
        # subset the u and v datasets 
        da_y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] )
        da_x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] )
        u_ds = nemo_u.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        v_ds = nemo_v.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        
        dr_n = np.where(np.diff(self.y_ind)>0, np.arange(0,u_ds.r_dim.size-1), np.nan )
        dr_n = dr_n[~np.isnan(dr_n)].astype(int)
        dr_s = np.where(np.diff(self.y_ind)<0, np.arange(0,u_ds.r_dim.size-1), np.nan )
        dr_s = dr_s[~np.isnan(dr_s)].astype(int)
        dr_e = np.where(np.diff(self.x_ind)>0, np.arange(0,v_ds.r_dim.size-1), np.nan )
        dr_e = dr_e[~np.isnan(dr_e)].astype(int)
        dr_w = np.where(np.diff(self.x_ind)<0, np.arange(0,v_ds.r_dim.size-1), np.nan )
        dr_w = dr_w[~np.isnan(dr_w)].astype(int)
        
        # Note that subsetting the dataset first instead of subsetting each array seperately,
        # as we do here, is neater but significantly slower.
        self.data_cross_flow['normal_velocities2'] = xr.full_like(u_ds.vozocrtx, np.nan)        
        self.data_cross_flow['normal_velocities2'][:,:,dr_n] = u_ds.vozocrtx.data[:,:,dr_n+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_s] = -u_ds.vozocrtx.data[:,:,dr_s]
        self.data_cross_flow['normal_velocities2'][:,:,dr_e] = -v_ds.vomecrty.data[:,:,dr_e+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_w] = v_ds.vomecrty.data[:,:,dr_w]
        self.data_cross_flow['normal_velocities2'].attrs = {'units':'m/s', \
                'standard_name':'contour-normal velocities'}
             
        self.data_cross_flow['normal_transport2'] = xr.full_like(u_ds.vozocrtx, np.nan)  
        self.data_cross_flow['normal_transport2'][:,:,dr_n] = ( u_ds.vozocrtx.data[:,:,dr_n+1] * 
                                u_ds.e2.data[dr_n+1] * u_ds.e3_0.data[:,dr_n+1] )
        self.data_cross_flow['normal_transport2'][:,:,dr_s] = ( -u_ds.vozocrtx.data[:,:,dr_s] * 
                                u_ds.e2.data[dr_s] * u_ds.e3_0.data[:,dr_s] )
        self.data_cross_flow['normal_transport2'][:,:,dr_e] = ( -v_ds.vomecrty.data[:,:,dr_e+1] *
                                v_ds.e1.data[dr_e+1] * v_ds.e3_0.data[:,dr_e+1] )
        self.data_cross_flow['normal_transport2'][:,:,dr_w] = ( v_ds.vomecrty.data[:,:,dr_w] *
                                v_ds.e1.data[dr_w] * v_ds.e3_0.data[:,dr_w] )
        self.data_cross_flow['normal_transport2'].attrs = {'units':'m^3/s', \
                'standard_name':'contour-normal volume transport'}
        
        self.data_cross_flow['depth_integrated_normal_transport2'] = (self.data_cross_flow
                                .normal_transport2.sum(dim='z_dim') / 1000000.)
        self.data_cross_flow['normal_transport2'].attrs ={'units':'Sv', \
                'standard_name':'contour-normal depth integrated volume transport'}
                                
        self.__update_cross_flow_vars('depth_0',u_ds.depth_0,v_ds.depth_0,dr_n,dr_s,dr_e,dr_w,1)
        self.__update_cross_flow_vars('longitude',u_ds.longitude,v_ds.longitude,dr_n,dr_s,dr_e,dr_w,0)
        self.__update_cross_flow_vars('latitude',u_ds.latitude,v_ds.latitude,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e1'] = xr.full_like(self.data_contour.e1, np.nan)   
        self.__update_cross_flow_vars('e1',u_ds.e1,v_ds.e1,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e2'] = xr.full_like(self.data_contour.e2, np.nan)
        self.__update_cross_flow_vars('e2',u_ds.e2,v_ds.e2,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e3_0'] = xr.full_like(self.data_contour.e3_0, np.nan)
        self.__update_cross_flow_vars('e3_0',u_ds.e3_0,v_ds.e3_0,dr_n,dr_s,dr_e,dr_w,1)
        
        self.data_cross_flow['depth_0'].attrs = {'standard_name':'Depth at time zero \
                on the contour-normal velocity grid points'}
        self.data_cross_flow['latitude'].attrs = {'standard_name':'Latitude at \
                the contour-normal velocity grid points'}
        self.data_cross_flow['longitude'].attrs = {'standard_name':'Longitude at \
                the contour-normal velocity grid points'}
                       
                     
        return
    
    def __update_cross_flow_vars(self, var, u_var,v_var, dr_n, dr_s, dr_e, dr_w, pos ):
        if pos==0:
            self.data_cross_flow[var][dr_n] = u_var.data[dr_n+1]
            self.data_cross_flow[var][dr_s] = u_var.data[dr_s]
            self.data_cross_flow[var][dr_e] = v_var.data[dr_e+1]
            self.data_cross_flow[var][dr_w] = v_var.data[dr_w]
            self.data_cross_flow[var][-1] = np.nan  
        elif pos==1:         
            self.data_cross_flow[var][:,dr_n] = u_var.data[:,dr_n+1]
            self.data_cross_flow[var][:,dr_s] = u_var.data[:,dr_s]
            self.data_cross_flow[var][:,dr_e] = v_var.data[:,dr_e+1]
            self.data_cross_flow[var][:,dr_w] = v_var.data[:,dr_w]
            self.data_cross_flow[var][:,-1] = np.nan    
        elif pos==2:         
            self.data_cross_flow[var][:,:,dr_n] = u_var.data[:,:,dr_n+1]
            self.data_cross_flow[var][:,:,dr_s] = u_var.data[:,:,dr_s]
            self.data_cross_flow[var][:,:,dr_e] = v_var.data[:,:,dr_e+1]
            self.data_cross_flow[var][:,:,dr_w] = v_var.data[:,:,dr_w]
            self.data_cross_flow[var][:,:,-1] = np.nan    
        
    
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
    
    
        def __pressure_grad_fpoint2(self, ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, r_ind, velocity_component):
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
            e2v = 0.5*( ds_T_j1.e2.data[r_ind] + ds_T.e2.data[r_ind] )
            e2v_i1 = 0.5*( ds_T_j1i1.e2.data[r_ind] + ds_T_i1.e2.data[r_ind] )
            e1v = 0.5*( ds_T_j1.e1.data[r_ind] + ds_T.e1.data[r_ind] )
            e1v_i1 = 0.5*( ds_T_j1i1.e1.data[r_ind] + ds_T_i1.e1.data[r_ind] )
            e1f = 0.5*( e1v + e1v_i1 )            
            # calculate gradients at v-points either side of f-point
            hpg = (ds_T_j1.pressure_h_zlevels.data[:,:,r_ind] - ds_T.pressure_h_zlevels.data[:,:,r_ind]) / e2v
            hpg_i1 = (ds_T_j1i1.pressure_h_zlevels.data[:,:,r_ind] - ds_T_i1.pressure_h_zlevels.data[:,:,r_ind]) / e2v_i1   
            # average onto f-point
            hpg_f = 0.5 * ( ( e1v * hpg ) + ( e1v_i1 * hpg_i1 ) ) / e1f 
            # as aboave            
            spg = (ds_T_j1.pressure_s.data[:,r_ind] - ds_T.pressure_s.data[:,r_ind]) / e2v
            spg_i1 = (ds_T_j1i1.pressure_s.data[:,r_ind] - ds_T_i1.pressure_s.data[:,r_ind]) / e2v_i1
            spg_f = 0.5 * ( (e1v * spg) + (e1v_i1 * spg_i1) ) / e1f 
        elif velocity_component == "v":
            # required scale factors for derivative and averaging
            e1u = 0.5 * ( ds_T_i1.e1.data[r_ind] + ds_T.e1.data[r_ind] ) 
            e1u_j1 = 0.5 * ( ds_T_j1i1.e1.data[r_ind] + ds_T_j1.e1.data[r_ind] )
            e2u = 0.5 * ( ds_T_i1.e2.data[r_ind] + ds_T.e2.data[r_ind] )
            e2u_j1 = 0.5 * ( ds_T_j1i1.e2.data[r_ind] + ds_T_j1.e2.data[r_ind] )
            e2f = 0.5 * ( e2u + e2u_j1 )
            # calculate gradients at u-points either side of f-point
            hpg = (ds_T_i1.pressure_h_zlevels.data[:,:,r_ind] - ds_T.pressure_h_zlevels.data[:,:,r_ind]) / e1u
            hpg_j1 = (ds_T_j1i1.pressure_h_zlevels.data[:,:,r_ind] - ds_T_j1.pressure_h_zlevels.data[:,:,r_ind]) / e1u_j1 
            # average onto f-point
            hpg_f = 0.5 * ( (e2u * hpg) + (e2u_j1 * hpg_j1) ) / e2f
            # as above
            spg = (ds_T_i1.pressure_s.data[:,r_ind] - ds_T.pressure_s.data[:,r_ind]) / e1u
            spg_j1 = (ds_T_j1i1.pressure_s.data[:,r_ind] - ds_T_j1.pressure_s.data[:,r_ind]) / e1u_j1 
            spg_f = 0.5 * ( (e2u * spg) + (e2u_j1 * spg_j1) ) / e2f
        
        return (hpg_f, spg_f)
   


    def calc_geostrophic_flow2(self, nemo_t: COAsT, ref_density = 1027):
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
        nemo_t_local = nemo_t.copy()
        if 't_dim' not in nemo_t_local.dataset.dims:
            nemo_t_local.dataset = nemo_t_local.dataset.expand_dims(dim={'t_dim':1},axis=0)

                       
   #     y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] ) # j
   #     x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] ) # i
        
        # We need to calculate the pressure at four t-points to get an
        # average onto the pressure gradient at the f-points, which will then
        # be averaged onto the normal velocity points. Here we subset the nemo_t 
        # data around the contour so we have these four t-grid points at each 
        # point along the contour        
        cont_t = Contour_t(nemo_t_local, self.y_ind, self.x_ind)            # j,i
        cont_t_j1 = Contour_t(nemo_t_local, self.y_ind+1, self.x_ind)       # j+1,i
        cont_t_i1 = Contour_t(nemo_t_local, self.y_ind, self.x_ind+1)       # j,i+1
        cont_t_j1i1 = Contour_t(nemo_t_local, self.y_ind+1, self.x_ind+1)   # j+1,i+1
        
        bath_max = np.max([cont_t.data_contour.bathymetry.max().item(), 
                           cont_t_j1.data_contour.bathymetry.max().item(),
                           cont_t_i1.data_contour.bathymetry.max().item(), 
                           cont_t_j1i1.data_contour.bathymetry.max().item()])   
        z_levels = self.gen_z_levels(bath_max)
        
        cont_t.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_j1.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_i1.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_j1i1.construct_pressure(1027, z_levels, extrapolate=True)        
        
        # Remove the mean hydrostatic pressure on each z_level from the hydrostatic pressure.
        # This helps to reduce the noise when taking the horizontal gradients of hydrostatic pressure.
        # Also catch and ignore nan-slice warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pressure_h_zlevel_mean = ( xr.concat( (cont_t.data_contour.pressure_h_zlevels, 
                            cont_t_j1.data_contour.pressure_h_zlevels, 
                            cont_t_i1.data_contour.pressure_h_zlevels, 
                            cont_t_j1i1.data_contour.pressure_h_zlevels), dim='concat_dim' )
                            .mean(dim=('concat_dim','r_dim','t_dim'),skipna=True) )
        cont_t.data_contour['pressure_h_zlevels'] = \
                cont_t.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_j1.data_contour['pressure_h_zlevels'] = \
                cont_t_j1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_i1.data_contour['pressure_h_zlevels'] = \
                cont_t_i1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_j1i1.data_contour['pressure_h_zlevels'] = \
                cont_t_j1i1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
                
        # Coriolis parameter
        f = 2 * 7.2921 * 10**(-5) * np.sin( np.deg2rad(self.data_contour.latitude) )
        
        
        dr_n = np.where(np.diff(self.y_ind)>0, np.arange(0,self.data_contour.r_dim.size-1), np.nan )
        dr_n = dr_n[~np.isnan(dr_n)].astype(int)
        dr_s = np.where(np.diff(self.y_ind)<0, np.arange(0,self.data_contour.r_dim.size-1), np.nan )
        dr_s = dr_s[~np.isnan(dr_s)].astype(int)
        dr_e = np.where(np.diff(self.x_ind)>0, np.arange(0,self.data_contour.r_dim.size-1), np.nan )
        dr_e = dr_e[~np.isnan(dr_e)].astype(int)
        dr_w = np.where(np.diff(self.x_ind)<0, np.arange(0,self.data_contour.r_dim.size-1), np.nan )
        dr_w = dr_w[~np.isnan(dr_w)].astype(int)
        
        
        hpg, spg = self.__pressure_grad_fpoint2( cont_t.data_contour, 
                        cont_t_j1.data_contour, cont_t_i1.data_contour, 
                        cont_t_j1i1.data_contour, dr_n 'u' )
        hpg_r1, spg_r1 = self.__pressure_grad_fpoint2( cont_t.data_contour,
                                        cont_t_j1.data_contour,
                                        cont_t_i1.data_contour, 
                                        cont_t_j1i1.data_contour, dr_n+1, "u" )
        
        # average from f to u point and calculate velocities
        e2u_j1 = 0.5 * (cont_t_j1.data_contour.e2.data[dr_n] + cont_t_j1i1.data_contour.e2.data[dr_n] )
        u_hpg = -(0.5 * (self.data_contour.e2.data[dr_n]*hpg/f.data[dr_n] + self.data_contour.e2.data[dr_n+1]*hpg_r1/f.data[dr_n+1]) 
                        / (e2u_j1 * ref_density))
        u_spg = -(0.5 * (self.data_contour.e2.data[dr_n]*spg/f.data[dr_n] + self.data_contour.e2.data[dr_n+1]*spg_r1/f.data[dr_n+1]) 
                        / (e2u_j1 * ref_density))                
        normal_velocity_hpg[:,:,dr_n] = u_hpg.values
        normal_velocity_spg[:,dr_n] = u_spg.values
        horizontal_scale[:,dr_n] = e2u_j1
        
        
        hpg, spg = self.__pressure_grad_fpoint2( cont_t.data_contour, 
                        cont_t_j1.data_contour, cont_t_i1.data_contour, 
                        cont_t_j1i1.data_contour, dr_s 'u' )
        hpg_r1, spg_r1 = self.__pressure_grad_fpoint2( cont_t.data_contour,
                                        cont_t_j1.data_contour,
                                        cont_t_i1.data_contour, 
                                        cont_t_j1i1.data_contour, dr_s+1, "u" )
                
        # average from f to u point and calculate velocities
        e2u = 0.5 * (cont_t.data_contour.e2.data[dr_s] + cont_t_i1.data_contour.e2.data[dr_s] )
        u_hpg = -(0.5 * (self.data_contour.e2.data[dr_s]*hpg/f.data[dr_s] + self.data_contour.e2.data[dr_s+1]*hpg_r1/f.data[dr_s+1]) 
                        / (e2u * ref_density))
        u_spg = -(0.5 * (self.data_contour.e2.data[dr_s]*spg/f.data[dr_s] + self.data_contour.e2.data[dr_s+1]*spg_r1/f.data[dr_s+1]) 
                        / (e2u * ref_density))                
        normal_velocity_hpg[:,:,dr_s] = -u_hpg.values
        normal_velocity_spg[:,dr_s] = -u_spg.values
        horizontal_scale[:,dr_s] = e2u
        
        
        # Note that subsetting the dataset first instead of subsetting each array seperately,
        # as we do here, is neater but the datasets need to be completely loaded into memory.
        self.data_cross_flow['normal_velocities2'] = xr.full_like(u_ds.vozocrtx, np.nan)        
        self.data_cross_flow['normal_velocities2'][:,:,dr_n] = u_ds.vozocrtx.data[:,:,dr_n+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_s] = -u_ds.vozocrtx.data[:,:,dr_s]
        self.data_cross_flow['normal_velocities2'][:,:,dr_e] = -v_ds.vomecrty.data[:,:,dr_e+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_w] = v_ds.vomecrty.data[:,:,dr_w]
        self.data_cross_flow['normal_velocities2'].attrs = {'units':'m/s', \
                'standard_name':'contour-normal velocities'}
        
        
            
        dy = np.diff(self.y_ind)
        dx = np.diff(self.x_ind)
        normal_velocity_hpg = np.zeros_like(cont_t.data_contour.pressure_h_zlevels)
        normal_velocity_spg = np.zeros_like(cont_t.data_contour.pressure_s)
        # horizontal scale factors for each segmant of contour
        horizontal_scale = np.zeros( (cont_t.data_contour.t_dim.size, cont_t.data_contour.r_dim.size) ) 
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):  
            # u flux (+u is positive across contour) 
            if dy[idx] > 0:    
                # calculate the pressure gradients at two f points defining a segment of the contour                             
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), 'u' )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "u" )
                
                # average from f to u point and calculate velocities
                e2u_j1 = 0.5 * (cont_t_j1.data_contour.isel(r_dim=idx).e2 + cont_t_j1i1.data_contour.isel(r_dim=idx).e2 )
                u_hpg = -(0.5 * (self.data_contour.e2[idx]*hpg/f[idx] + self.data_contour.e2[idx+1]*hpg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))
                u_spg = -(0.5 * (self.data_contour.e2[idx]*spg/f[idx] + self.data_contour.e2[idx+1]*spg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))                
                normal_velocity_hpg[:,:,idx] = u_hpg.values
                normal_velocity_spg[:,idx] = u_spg.values
                horizontal_scale[:,idx] = e2u_j1
             
            # u flux (-u is positive across contour)     
            elif dy[idx] < 0:
                # calculate the pressure gradients at two f points defining a segment of the contour                             
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), 'u' )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "u" )
                
                # average from f to u point and calculate velocities
                e2u = 0.5 * (cont_t.data_contour.isel(r_dim=idx).e2 + cont_t_i1.data_contour.isel(r_dim=idx).e2 )
                u_hpg = -(0.5 * (self.data_contour.e2[idx]*hpg/f[idx] + self.data_contour.e2[idx+1]*hpg_r1/f[idx+1]) 
                                / (e2u * ref_density))
                u_spg = -(0.5 * (self.data_contour.e2[idx]*spg/f[idx] + self.data_contour.e2[idx+1]*spg_r1/f[idx+1]) 
                                / (e2u * ref_density))                
                normal_velocity_hpg[:,:,idx] = -u_hpg.values
                normal_velocity_spg[:,idx] = -u_spg.values
                horizontal_scale[:,idx] = e2u
             
            # v flux (-v is positive across contour)
            elif dx[idx] > 0: 
                # calculate the pressure gradients at two f points defining a segment of the contour                                  
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), "v" )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "v" )
                
                # average from f to v point and calculate velocities
                e1v_i1 = 0.5 * ( cont_t_i1.data_contour.isel(r_dim=idx).e1 + cont_t_j1i1.data_contour.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_contour.e1[idx]*hpg/f[idx] + self.data_contour.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                v_spg = (0.5 * (self.data_contour.e1[idx]*spg/f[idx] + self.data_contour.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                normal_velocity_hpg[:,:,idx] = -v_hpg.values
                normal_velocity_spg[:,idx] = -v_spg.values
                horizontal_scale[:,idx] = e1v_i1
                
            # v flux (+v is positive across contour)    
            elif dx[idx] < 0:
                # calculate the pressure gradients at two f points defining a segment of the contour                                  
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), "v" )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "v" )
                
                # average from f to v point and calculate velocities
                e1v = 0.5 * ( cont_t.data_contour.isel(r_dim=idx).e1 + cont_t_j1.data_contour.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_contour.e1[idx]*hpg/f[idx] + self.data_contour.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v * ref_density))
                v_spg = (0.5 * (self.data_contour.e1[idx]*spg/f[idx] + self.data_contour.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v * ref_density))                   
                normal_velocity_hpg[:,:,idx] = v_hpg.values 
                normal_velocity_spg[:,idx] = v_spg.values 
                horizontal_scale[:,idx] = e1v
        
        normal_velocity_hpg[:,:,-1] = np.nan
        normal_velocity_spg[:,-1] = np.nan
        
        H = np.zeros_like( self.data_contour.bathymetry.values )
        H[:-1] = 0.5*(self.data_contour.bathymetry.values[:-1] + self.data_contour.bathymetry.values[1:])
        normal_velocity_hpg = np.where( z_levels[:,np.newaxis] <= H, 
                               normal_velocity_hpg, np.nan )
        
        # remove redundent levels    
        active_z_levels = np.count_nonzero(~np.isnan(normal_velocity_hpg),axis=1).max() 
        normal_velocity_hpg = normal_velocity_hpg[:,:active_z_levels,:]
        z_levels = cont_t.data_contour.depth_z_levels.values[:active_z_levels]
        
        coords_hpg={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), self.data_contour.latitude),
                'longitude': (('r_dim'), self.data_contour.longitude)}
        dims_hpg=['depth_z_levels', 'r_dim']
        attributes_hpg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the hydrostatic pressure gradient'}
        coords_spg={'latitude': (('r_dim'), self.data_contour.latitude),
                'longitude': (('r_dim'), self.data_contour.longitude)}
        dims_spg=['r_dim']
        attributes_spg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the surface pressure gradient'}
        
        if 't_dim' in cont_t.data_contour.dims:
            coords_hpg['time'] = (('t_dim'), cont_t.data_contour.time)
            dims_hpg.insert(0, 't_dim')
            coords_spg['time'] = (('t_dim'), cont_t.data_contour.time)
            dims_spg.insert(0, 't_dim')
        
        self.data_cross_flow['normal_velocity_hpg'] = xr.DataArray( np.squeeze(normal_velocity_hpg),
                coords=coords_hpg, dims=dims_hpg, attrs=attributes_hpg)
        self.data_cross_flow['normal_velocity_spg'] = xr.DataArray( np.squeeze(normal_velocity_spg),
                coords=coords_spg, dims=dims_spg, attrs=attributes_spg)

        self.data_cross_flow['transport_across_AB_hpg'] = ( self.data_cross_flow
                .normal_velocity_hpg.fillna(0).integrate(dim='depth_z_levels') ) * horizontal_scale / 1000000   
        self.data_cross_flow.transport_across_AB_hpg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the hydrostatic pressure gradient'}
        
        
        #depth_3d = self.data_tran.depth_z_levels.broadcast_like(self.data_tran.normal_velocity_hpg)
        #H = depth_3d.where(~self.data_tran.normal_velocity_hpg.to_masked_array().mask).max(dim='z_dim')
        #H = 0.5*(self.data_contour.bathymetry.values + self.data_contour.bathymetry[1:].values)
        self.data_cross_flow['transport_across_AB_spg'] = self.data_cross_flow.normal_velocity_spg * H * horizontal_scale / 1000000
        self.data_cross_flow.transport_across_AB_spg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the surface pressure gradient'}
        
        return

   
    def calc_geostrophic_flow(self, nemo_t: COAsT, ref_density = 1027):
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
        nemo_t_local = nemo_t.copy()
        if 't_dim' not in nemo_t_local.dataset.dims:
            nemo_t_ds = nemo_t_local.dataset.expand_dims(dim={'t_dim':1},axis=0)
        else:
            nemo_t_ds = nemo_t_local.dataset
                       
   #     y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] ) # j
   #     x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] ) # i
        
        # We need to calculate the pressure at four t-points to get an
        # average onto the pressure gradient at the f-points, which will then
        # be averaged onto the normal velocity points. Here we subset the nemo_t 
        # data around the contour so we have these four t-grid points at each 
        # point along the contour        
        cont_t = Contour_t(nemo_t_local, self.y_ind, self.x_ind)            # j,i
        cont_t_j1 = Contour_t(nemo_t_local, self.y_ind+1, self.x_ind)       # j+1,i
        cont_t_i1 = Contour_t(nemo_t_local, self.y_ind, self.x_ind+1)       # j,i+1
        cont_t_j1i1 = Contour_t(nemo_t_local, self.y_ind+1, self.x_ind+1)   # j+1,i+1
        
        bath_max = np.max([cont_t.data_contour.bathymetry.max().item(), 
                           cont_t_j1.data_contour.bathymetry.max().item(),
                           cont_t_i1.data_contour.bathymetry.max().item(), 
                           cont_t_j1i1.data_contour.bathymetry.max().item()])   
        z_levels = self.gen_z_levels(bath_max)
        
        cont_t.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_j1.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_i1.construct_pressure(1027, z_levels, extrapolate=True)
        cont_t_j1i1.construct_pressure(1027, z_levels, extrapolate=True)        
        
        # Remove the mean hydrostatic pressure on each z_level from the hydrostatic pressure.
        # This helps to reduce the noise when taking the horizontal gradients of hydrostatic pressure.
        # Also catch and ignore nan-slice warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pressure_h_zlevel_mean = ( xr.concat( (cont_t.data_contour.pressure_h_zlevels, 
                            cont_t_j1.data_contour.pressure_h_zlevels, 
                            cont_t_i1.data_contour.pressure_h_zlevels, 
                            cont_t_j1i1.data_contour.pressure_h_zlevels), dim='concat_dim' )
                            .mean(dim=('concat_dim','r_dim','t_dim'),skipna=True) )
        cont_t.data_contour['pressure_h_zlevels'] = \
                cont_t.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_j1.data_contour['pressure_h_zlevels'] = \
                cont_t_j1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_i1.data_contour['pressure_h_zlevels'] = \
                cont_t_i1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
        cont_t_j1i1.data_contour['pressure_h_zlevels'] = \
                cont_t_j1i1.data_contour.pressure_h_zlevels - pressure_h_zlevel_mean
                
        # Coriolis parameter
        f = 2 * 7.2921 * 10**(-5) * np.sin( np.deg2rad(self.data_contour.latitude) )
        
        dy = np.diff(self.y_ind)
        dx = np.diff(self.x_ind)
        normal_velocity_hpg = np.zeros_like(cont_t.data_contour.pressure_h_zlevels)
        normal_velocity_spg = np.zeros_like(cont_t.data_contour.pressure_s)
        # horizontal scale factors for each segmant of contour
        horizontal_scale = np.zeros( (cont_t.data_contour.t_dim.size, cont_t.data_contour.r_dim.size) ) 
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):  
            # u flux (+u is positive across contour) 
            if dy[idx] > 0:    
                # calculate the pressure gradients at two f points defining a segment of the contour                             
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), 'u' )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "u" )
                
                # average from f to u point and calculate velocities
                e2u_j1 = 0.5 * (cont_t_j1.data_contour.isel(r_dim=idx).e2 + cont_t_j1i1.data_contour.isel(r_dim=idx).e2 )
                u_hpg = -(0.5 * (self.data_contour.e2[idx]*hpg/f[idx] + self.data_contour.e2[idx+1]*hpg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))
                u_spg = -(0.5 * (self.data_contour.e2[idx]*spg/f[idx] + self.data_contour.e2[idx+1]*spg_r1/f[idx+1]) 
                                / (e2u_j1 * ref_density))                
                normal_velocity_hpg[:,:,idx] = u_hpg.values
                normal_velocity_spg[:,idx] = u_spg.values
                horizontal_scale[:,idx] = e2u_j1
             
            # u flux (-u is positive across contour)     
            elif dy[idx] < 0:
                # calculate the pressure gradients at two f points defining a segment of the contour                             
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), 'u' )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "u" )
                
                # average from f to u point and calculate velocities
                e2u = 0.5 * (cont_t.data_contour.isel(r_dim=idx).e2 + cont_t_i1.data_contour.isel(r_dim=idx).e2 )
                u_hpg = -(0.5 * (self.data_contour.e2[idx]*hpg/f[idx] + self.data_contour.e2[idx+1]*hpg_r1/f[idx+1]) 
                                / (e2u * ref_density))
                u_spg = -(0.5 * (self.data_contour.e2[idx]*spg/f[idx] + self.data_contour.e2[idx+1]*spg_r1/f[idx+1]) 
                                / (e2u * ref_density))                
                normal_velocity_hpg[:,:,idx] = -u_hpg.values
                normal_velocity_spg[:,idx] = -u_spg.values
                horizontal_scale[:,idx] = e2u
             
            # v flux (-v is positive across contour)
            elif dx[idx] > 0: 
                # calculate the pressure gradients at two f points defining a segment of the contour                                  
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), "v" )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "v" )
                
                # average from f to v point and calculate velocities
                e1v_i1 = 0.5 * ( cont_t_i1.data_contour.isel(r_dim=idx).e1 + cont_t_j1i1.data_contour.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_contour.e1[idx]*hpg/f[idx] + self.data_contour.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                v_spg = (0.5 * (self.data_contour.e1[idx]*spg/f[idx] + self.data_contour.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v_i1 * ref_density))
                normal_velocity_hpg[:,:,idx] = -v_hpg.values
                normal_velocity_spg[:,idx] = -v_spg.values
                horizontal_scale[:,idx] = e1v_i1
                
            # v flux (+v is positive across contour)    
            elif dx[idx] < 0:
                # calculate the pressure gradients at two f points defining a segment of the contour                                  
                hpg, spg = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx), 
                                        cont_t_j1.data_contour.isel(r_dim=idx),
                                        cont_t_i1.data_contour.isel(r_dim=idx), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx), "v" )
                hpg_r1, spg_r1 = self.__pressure_grad_fpoint( cont_t.data_contour.isel(r_dim=idx+1),
                                        cont_t_j1.data_contour.isel(r_dim=idx+1),
                                        cont_t_i1.data_contour.isel(r_dim=idx+1), 
                                        cont_t_j1i1.data_contour.isel(r_dim=idx+1), "v" )
                
                # average from f to v point and calculate velocities
                e1v = 0.5 * ( cont_t.data_contour.isel(r_dim=idx).e1 + cont_t_j1.data_contour.isel(r_dim=idx).e1 )
                v_hpg = (0.5 * (self.data_contour.e1[idx]*hpg/f[idx] + self.data_contour.e1[idx+1]*hpg_r1/f[idx+1])
                                / (e1v * ref_density))
                v_spg = (0.5 * (self.data_contour.e1[idx]*spg/f[idx] + self.data_contour.e1[idx+1]*spg_r1/f[idx+1])
                                / (e1v * ref_density))                   
                normal_velocity_hpg[:,:,idx] = v_hpg.values 
                normal_velocity_spg[:,idx] = v_spg.values 
                horizontal_scale[:,idx] = e1v
        
        normal_velocity_hpg[:,:,-1] = np.nan
        normal_velocity_spg[:,-1] = np.nan
        
        H = np.zeros_like( self.data_contour.bathymetry.values )
        H[:-1] = 0.5*(self.data_contour.bathymetry.values[:-1] + self.data_contour.bathymetry.values[1:])
        normal_velocity_hpg = np.where( z_levels[:,np.newaxis] <= H, 
                               normal_velocity_hpg, np.nan )
        
        # remove redundent levels    
        active_z_levels = np.count_nonzero(~np.isnan(normal_velocity_hpg),axis=1).max() 
        normal_velocity_hpg = normal_velocity_hpg[:,:active_z_levels,:]
        z_levels = cont_t.data_contour.depth_z_levels.values[:active_z_levels]
        
        coords_hpg={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), self.data_contour.latitude),
                'longitude': (('r_dim'), self.data_contour.longitude)}
        dims_hpg=['depth_z_levels', 'r_dim']
        attributes_hpg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the hydrostatic pressure gradient'}
        coords_spg={'latitude': (('r_dim'), self.data_contour.latitude),
                'longitude': (('r_dim'), self.data_contour.longitude)}
        dims_spg=['r_dim']
        attributes_spg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the surface pressure gradient'}
        
        if 't_dim' in cont_t.data_contour.dims:
            coords_hpg['time'] = (('t_dim'), cont_t.data_contour.time)
            dims_hpg.insert(0, 't_dim')
            coords_spg['time'] = (('t_dim'), cont_t.data_contour.time)
            dims_spg.insert(0, 't_dim')
        
        self.data_cross_flow['normal_velocity_hpg'] = xr.DataArray( np.squeeze(normal_velocity_hpg),
                coords=coords_hpg, dims=dims_hpg, attrs=attributes_hpg)
        self.data_cross_flow['normal_velocity_spg'] = xr.DataArray( np.squeeze(normal_velocity_spg),
                coords=coords_spg, dims=dims_spg, attrs=attributes_spg)

        self.data_cross_flow['transport_across_AB_hpg'] = ( self.data_cross_flow
                .normal_velocity_hpg.fillna(0).integrate(dim='depth_z_levels') ) * horizontal_scale / 1000000   
        self.data_cross_flow.transport_across_AB_hpg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the hydrostatic pressure gradient'}
        
        
        #depth_3d = self.data_tran.depth_z_levels.broadcast_like(self.data_tran.normal_velocity_hpg)
        #H = depth_3d.where(~self.data_tran.normal_velocity_hpg.to_masked_array().mask).max(dim='z_dim')
        #H = 0.5*(self.data_contour.bathymetry.values + self.data_contour.bathymetry[1:].values)
        self.data_cross_flow['transport_across_AB_spg'] = self.data_cross_flow.normal_velocity_spg * H * horizontal_scale / 1000000
        self.data_cross_flow.transport_across_AB_spg.attrs = {'units': 'Sv', 
                                  'standard_name': 'volume transport across transect due to the surface pressure gradient'}
        
        return
    
    # def transport_across_AB(self, nemo_u: COAsT, nemo_v: COAsT):
    #     """
    
    #     Computes the flow across the contour at each segment and stores:
    #     Transect normal velocities at each grid point in Transect.normal_velocities,
    #     Depth integrated volume transport across the transect at each transect segment in 
    #     Transect.depth_integrated_transport_across_AB
        
    #     Return 
    #     -----------
    #     Transect normal velocities at each grid point (m/s)
    #     Depth integrated volume transport across the transect at each transect segment (Sv)
    #     """
        
    #     # subset the u and v datasets 
    #     da_y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] )
    #     da_x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] )
    #     u_ds = nemo_u.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
    #     v_ds = nemo_v.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        
    #     velocity = np.ma.zeros(np.shape(u_ds.vozocrtx))
    #     vol_transport = np.ma.zeros(np.shape(u_ds.vozocrtx))
    #     depth_integrated_transport = np.ma.zeros( np.shape(u_ds.vozocrtx[:,0,:] )) 
    #     #depth_0 = np.ma.zeros(np.shape(self.data_U.depth_0))
 
    #     dy = np.diff(self.y_ind)
    #     dx = np.diff(self.x_ind)
    #     # Loop through each point along the transact
    #     for idx in np.arange(0, self.len-1):            
    #         if dy[idx] > 0:
    #             # u flux (+ in)
    #             velocity[:,:,idx] = u_ds.vozocrtx[:,:,idx+1].to_masked_array()
    #             vol_transport[:,:,idx] = ( velocity[:,:,idx] * u_ds.e2[idx+1].to_masked_array() *
    #                                       u_ds.e3_0[:,idx+1].to_masked_array() )
    #             depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
    #             #depth_0[:,idx] = self.data_U.depth_0[:,idx+1].to_masked_array()
    #         elif dy[idx] < 0:
    #             # u flux (-u is positive across contour) 
    #             velocity[:,:,idx] = - u_ds.vozocrtx[:,:,idx].to_masked_array()
    #             vol_transport[:,:,idx] = ( velocity[:,:,idx] * u_ds.e2[idx].to_masked_array() *
    #                                       u_ds.e3_0[:,idx].to_masked_array() )
    #             depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
    #             #depth_0[:,idx] = self.data_U.depth_0[:,idx].to_masked_array()
    #         elif dx[idx] > 0:
    #             # v flux (- in) 
    #             velocity[:,:,idx] = - v_ds.vomecrty[:,:,idx+1].to_masked_array()
    #             vol_transport[:,:,idx] = ( velocity[:,:,idx] * v_ds.e1[idx+1].to_masked_array() *
    #                                       v_ds.e3_0[:,idx+1].to_masked_array() )
    #             depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
    #             #depth_0[:,idx] = self.data_V.depth_0[:,idx+1].to_masked_array()
    #         elif dx[idx] < 0:
    #             # v flux (+ in)
    #             velocity[:,:,idx] = v_ds.vomecrty[:,:,idx].to_masked_array()
    #             vol_transport[:,:,idx] = ( velocity[:,:,idx] * v_ds.e1[idx].to_masked_array() *
    #                                       v_ds.e3_0[:,idx].to_masked_array() )
    #             depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
    #             #depth_0[:,idx] = self.data_V.depth_0[:,idx].to_masked_array()
        
        
    #     velocity[:,:,-1] = np.nan
    #     depth_integrated_transport[:,-1] = np.nan
        
    #     self.data_cross_flow['normal_velocities'] = xr.DataArray( velocity, 
    #                 coords={'time': (('t_dim'), u_ds.time.values),'depth_0': (('z_dim','r_dim'), self.data_contour.depth_0)},
    #                 dims=['t_dim', 'z_dim', 'r_dim'] )        
    
    #     self.data_cross_flow['depth_integrated_transport_across_AB'] = xr.DataArray( depth_integrated_transport / 1000000.,
    #                 coords={'time': (('t_dim'), u_ds.time.values)},
    #                 dims=['t_dim', 'r_dim'] ) 
        
    #     #self.data_cross_flow.depth_0.attrs['units'] = 'm'
    #     #self.data_cross_flow.depth_0.attrs['standard_name'] = 'Initial depth at time zero'
    #     #self.data_cross_flow.depth_0.attrs['long_name'] = 'Initial depth at time zero defined at the normal velocity grid points on the transect'
                        
    #     return u_ds, v_ds


class Contour_t(Contour):
    def __init__(self, nemo_t: COAsT, y_ind, x_ind, depth=None):
        super().__init__(nemo_t, y_ind, x_ind, depth)
        
        
    def construct_pressure( self, ref_density, z_levels=None, extrapolate=False ):   
        '''
            This method is for calculating the hydrostatic and surface pressure fields
            on z-levels. The motivation is to enable the calculation of horizontal 
            gradients; however, the variables can quite easily be interpolated 
            onto the original vertical grid (which is a less expensive operation)
             
            Requirements: The object's t-grid dataset must contain the sea surface height,
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
            
            This code could be rewritten to make better use of xarray

        Parameters
        ----------
        ref_density
            reference density value
        z_levels : (optional) numpy array
            1d array that defines the depths to interpolate the density and pressure
            on to.
        extrapolate : boolean, default False
            If true the variables are extrapolated to the deepest z_levels, if false
            values below the bathymetry are set to NaN
        Returns
        -------
        None.

        '''        

        if 't_dim' not in self.data_contour.dims:
            self.data_contour = self.data_contour.expand_dims(dim={'t_dim':1},axis=0)

        if z_levels is None:   
            # #bathymetry = xr.open_dataset( self.filename_domain ).bathy_metry.squeeze()
            # z_levels_0_50 = np.arange(0,55,5.5)
            # z_levels_60_200 = np.arange(60,210,10)
            # z_levels_250_600 = np.arange(200,650,50)
            # #z_levels_650_ = np.arange(650,bathymetry.max()+150,150)
            # z_levels_650_ = np.arange(650,self.data_contour.bathymetry.max().item()+150,150)
            # z_levels = np.concatenate( (z_levels_0_50, z_levels_60_200, 
            #                             z_levels_250_600, z_levels_650_) )
            z_levels = self.gen_z_levels( self.data_contour.bathymetry.max().item() )
        
        shape_ds = ( self.data_contour.t_dim.size, len(z_levels), self.data_contour.r_dim.size )
        salinity_z = np.ma.zeros( shape_ds )
        temperature_z = np.ma.zeros( shape_ds ) 
        salinity_s = self.data_contour.salinity.to_masked_array()
        temperature_s = self.data_contour.temperature.to_masked_array()
        s_levels = self.data_contour.depth_0.values
        
        # At the current time there does not appear to be a good algorithm for performing this 
        # type of interpolation without loops, which is very slow. Griddata is an option but does not
        # support extrapolation and did not have noticable performance benefits.
        for it in self.data_contour.t_dim:
            for ir in self.data_contour.r_dim:
                if not np.all(np.isnan(salinity_s[it,:,ir].data)):  
                    # Need to remove the levels below the (envelope) bathymetry which are NaN
                    salinity_s_r = salinity_s[it,:,ir].compressed()
                    temperature_s_r = temperature_s[it,:,ir].compressed()
                    s_levels_r = s_levels[:len(salinity_s_r),ir]
                    
                    sal_func = interpolate.interp1d( s_levels_r, salinity_s_r, 
                                 kind='linear', fill_value="extrapolate")
                    temp_func = interpolate.interp1d( s_levels_r, temperature_s_r, 
                                 kind='linear', fill_value="extrapolate")
                    
                    if extrapolate is True:
                        salinity_z[it,:,ir] = sal_func(z_levels)
                        temperature_z[it,:,ir] = temp_func(z_levels)                        
                    else:
                        # set levels below the bathymetry to nan
                        salinity_z[it,:,ir] = np.where( z_levels <= self.data_contour.bathymetry.values[ir], 
                                sal_func(z_levels), np.nan )
                        temperature_z[it,:,ir] = np.where( z_levels <= self.data_contour.bathymetry.values[ir], 
                                temp_func(z_levels), np.nan ) 
                    
        if extrapolate is False:
            # remove redundent levels    
            active_z_levels = np.count_nonzero(~np.isnan(salinity_z),axis=1).max() 
            salinity_z = salinity_z[:,:active_z_levels,:]
            temperature_z = temperature_z[:,:active_z_levels,:]
            z_levels = z_levels[:active_z_levels]
        
        # Absolute Pressure    
        pressure_absolute = np.ma.masked_invalid(
            gsw.p_from_z( -z_levels[:,np.newaxis], self.data_contour.latitude ) ) # depth must be negative           
        # Absolute Salinity           
        salinity_absolute = np.ma.masked_invalid(
            gsw.SA_from_SP( salinity_z, pressure_absolute, self.data_contour.longitude, self.data_contour.latitude ) )
        salinity_absolute = np.ma.masked_less(salinity_absolute,0)
        # Conservative Temperature
        temp_conservative = np.ma.masked_invalid(
            gsw.CT_from_pt( salinity_absolute, temperature_z ) )
        # In-situ density
        density_z = np.ma.masked_invalid( gsw.rho( 
            salinity_absolute, temp_conservative, pressure_absolute ) )
        
                        
        coords={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), self.data_contour.latitude),
                'longitude': (('r_dim'), self.data_contour.longitude)}
        dims=['depth_z_levels', 'r_dim']
        attributes = {'units': 'kg / m^3', 'standard name': 'In-situ density on the z-level vertical grid'}
        
        if shape_ds[0] != 1:
            coords['time'] = (('t_dim'), self.data_contour.time.values)
            dims.insert(0, 't_dim')
          
        self.data_contour['density_zlevels'] = xr.DataArray( np.squeeze(density_z), 
                coords=coords, dims=dims, attrs=attributes )
    
        # cumulative integral of density on z levels
        # Note that zero density flux is assumed at z=0
        density_cumulative = -cumtrapz( density_z, x=-z_levels, axis=1, initial=0)

        hydrostatic_pressure = density_cumulative * self.GRAVITY
        
        attributes = {'units': 'kg m^{-1} s^{-2}', 'standard name': 'Hydrostatic pressure on the z-level vertical grid'}
        self.data_contour['pressure_h_zlevels'] = xr.DataArray( np.squeeze(hydrostatic_pressure), 
                coords=coords, dims=dims, attrs=attributes )
        
        self.data_contour['pressure_s'] = ref_density * self.GRAVITY * self.data_contour.ssh.squeeze()
        self.data_contour.pressure_s.attrs = {'units': 'kg m^{-1} s^{-2}', 
                                  'standard_name': 'surface pressure'}
        
        return
    
        
        
    