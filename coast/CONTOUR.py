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
    
    def __init__(self, nemo: COAsT, y_ind, x_ind):        
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
        y_ind = y_ind[nonrepeated_idx]
        x_ind = x_ind[nonrepeated_idx]
        
        if y_ind[0] > y_ind[-1]:
            y_ind = y_ind[::-1]
            x_ind = x_ind[::-1]
        return (y_ind, x_ind)
    

class Contour_f(Contour):
    def __init__(self, nemo_f: COAsT, y_ind, x_ind):
        super().__init__(nemo_f, y_ind, x_ind)
        self.data_cross_flow = xr.Dataset()
        
    def transport_across_AB_2(self, nemo_u: COAsT, nemo_v: COAsT):
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
        
        self.data_cross_flow['normal_velocities2'] = xr.full_like(u_ds.vozocrtx, np.nan)        
        self.data_cross_flow['normal_velocities2'][:,:,dr_n] = u_ds.vozocrtx.data[:,:,dr_n+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_s] = -u_ds.vozocrtx.data[:,:,dr_s]
        self.data_cross_flow['normal_velocities2'][:,:,dr_e] = -v_ds.vomecrty.data[:,:,dr_e+1]
        self.data_cross_flow['normal_velocities2'][:,:,dr_w] = v_ds.vomecrty.data[:,:,dr_w]
        self.data_cross_flow['normal_velocities2'].attrs({'units':'m/s', \
                'standard_name':'contour-normal velocities'})
             
        self.data_cross_flow['normal_transport2'] = xr.full_like(u_ds.vozocrtx, np.nan)  
        self.data_cross_flow['normal_transport2'][:,:,dr_n] = ( u_ds.vozocrtx.data[:,:,dr_n+1] * 
                                u_ds.e2.data[dr_n+1] * u_ds.e3_0.data[:,dr_n+1] )
        self.data_cross_flow['normal_transport2'][:,:,dr_s] = ( -u_ds.vozocrtx.data[:,:,dr_s] * 
                                u_ds.e2.data[dr_s] * u_ds.e3_0.data[:,dr_s] )
        self.data_cross_flow['normal_transport2'][:,:,dr_e] = ( -v_ds.vomecrty.data[:,:,dr_e+1] *
                                v_ds.e1.data[dr_e+1] * v_ds.e3_0.data[:,dr_e+1] )
        self.data_cross_flow['normal_transport2'][:,:,dr_w] = ( v_ds.vomecrty.data[:,:,dr_w] *
                                v_ds.e1.data[dr_w] * v_ds.e3_0.data[:,dr_w] )
        self.data_cross_flow['normal_transport2'].attrs({'units':'m^3/s', \
                'standard_name':'contour-normal volume transport'})
        
        self.data_cross_flow['depth_integrated_normal_transport2'] = (self.data_cross_flow
                                .normal_transport2.sum(dim='z_dim') / 1000000.)
        self.data_cross_flow['normal_transport2'].attrs({'units':'Sv', \
                'standard_name':'contour-normal depth integrated volume transport'})
                                
        self.__update_cross_flow_vars('depth_0',u_ds.depth_0,v_ds.depth_0,dr_n,dr_s,dr_e,dr_w,1)
        self.__update_cross_flow_vars('longitude',u_ds.longitude,v_ds.longitude,dr_n,dr_s,dr_e,dr_w,0)
        self.__update_cross_flow_vars('latitude',u_ds.latitude,v_ds.latitude,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e1'] = xr.full_like(self.data_contour.e1, np.nan)   
        self.__update_cross_flow_vars('e1',u_ds.e1,v_ds.e1,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e2'] = xr.full_like(self.data_contour.e2, np.nan)
        self.__update_cross_flow_vars('e2',u_ds.e2,v_ds.e2,dr_n,dr_s,dr_e,dr_w,0)
        self.data_cross_flow['e3_0'] = xr.full_like(self.data_contour.e3_0, np.nan)
        self.__update_cross_flow_vars('e3_0',u_ds.e3_0,v_ds.e3_0,dr_n,dr_s,dr_e,dr_w,1)
        
        self.data_cross_flow['depth_0'].attrs({'standard_name':'Depth at time zero \
                on the contour-normal velocity grid points'})
        self.data_cross_flow['latitude'].attrs({'standard_name':'Latitude at \
                the contour-normal velocity grid points'})
        self.data_cross_flow['longitude'].attrs({'standard_name':'Longitude at \
                the contour-normal velocity grid points'})
                       
                     
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
        
    
    
    def transport_across_AB(self, nemo_u: COAsT, nemo_v: COAsT):
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
        
        velocity = np.ma.zeros(np.shape(u_ds.vozocrtx))
        vol_transport = np.ma.zeros(np.shape(u_ds.vozocrtx))
        depth_integrated_transport = np.ma.zeros( np.shape(u_ds.vozocrtx[:,0,:] )) 
        #depth_0 = np.ma.zeros(np.shape(self.data_U.depth_0))
 
        dy = np.diff(self.y_ind)
        dx = np.diff(self.x_ind)
        # Loop through each point along the transact
        for idx in np.arange(0, self.len-1):            
            if dy[idx] > 0:
                # u flux (+ in)
                velocity[:,:,idx] = u_ds.vozocrtx[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * u_ds.e2[idx+1].to_masked_array() *
                                          u_ds.e3_0[:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                #depth_0[:,idx] = self.data_U.depth_0[:,idx+1].to_masked_array()
            elif dy[idx] < 0:
                # u flux (-u is positive across contour) 
                velocity[:,:,idx] = - u_ds.vozocrtx[:,:,idx].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * u_ds.e2[idx].to_masked_array() *
                                          u_ds.e3_0[:,idx].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                #depth_0[:,idx] = self.data_U.depth_0[:,idx].to_masked_array()
            elif dx[idx] > 0:
                # v flux (- in) 
                velocity[:,:,idx] = - v_ds.vomecrty[:,:,idx+1].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * v_ds.e1[idx+1].to_masked_array() *
                                          v_ds.e3_0[:,idx+1].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                #depth_0[:,idx] = self.data_V.depth_0[:,idx+1].to_masked_array()
            elif dx[idx] < 0:
                # v flux (+ in)
                velocity[:,:,idx] = v_ds.vomecrty[:,:,idx].to_masked_array()
                vol_transport[:,:,idx] = ( velocity[:,:,idx] * v_ds.e1[idx].to_masked_array() *
                                          v_ds.e3_0[:,idx].to_masked_array() )
                depth_integrated_transport[:,idx] = np.sum( vol_transport[:,:,idx], axis=1 )
                #depth_0[:,idx] = self.data_V.depth_0[:,idx].to_masked_array()
        
        
        velocity[:,:,-1] = np.nan
        depth_integrated_transport[:,-1] = np.nan
        
        self.data_cross_flow['normal_velocities'] = xr.DataArray( velocity, 
                    coords={'time': (('t_dim'), u_ds.time.values),'depth_0': (('z_dim','r_dim'), self.data_contour.depth_0)},
                    dims=['t_dim', 'z_dim', 'r_dim'] )        
    
        self.data_cross_flow['depth_integrated_transport_across_AB'] = xr.DataArray( depth_integrated_transport / 1000000.,
                    coords={'time': (('t_dim'), u_ds.time.values)},
                    dims=['t_dim', 'r_dim'] ) 
        
        #self.data_cross_flow.depth_0.attrs['units'] = 'm'
        #self.data_cross_flow.depth_0.attrs['standard_name'] = 'Initial depth at time zero'
        #self.data_cross_flow.depth_0.attrs['long_name'] = 'Initial depth at time zero defined at the normal velocity grid points on the transect'
                        
        return u_ds, v_ds

        
        
    