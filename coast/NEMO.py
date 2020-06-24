from .COAsT import COAsT
import xarray as xa
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def set_dimension_mapping(self):
        #self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim', 
        #                    'y':'y_dim', 'x':'x_dim'}
        self.dim_mapping = None
        
    def set_variable_mapping(self):
        #self.var_mapping = {'time_counter':'time',
        #                    'votemper' : 'temperature',
        #                    'temp' : 'temperature'}
        self.var_mapping = None

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller
    
class Transect:
    def __init__(self, dataset: xa.Dataset, domain: COAsT, point_A: tuple, point_B: tuple):
        
        self.point_A = point_A
        self.point_B = point_B
        # Get points on transect
        tran_y, tran_x, tran_len = domain.transect_indices(point_A, point_B, grid_ref="f")
        tran_y = np.asarray(tran_y)
        tran_x = np.asarray(tran_x)
        
        # Redefine transect to be defined as a 'sawtooth' transect, i.e. each point on
        # the transect is seperated from its neighbours by a single index change in y or x, but not both
        dist_option_1 = domain.e2f.values[0, tran_y, tran_x] + domain.e1f.values[0, tran_y+1, tran_x]
        dist_option_2 = domain.e2f.values[0, tran_y, tran_x+1] + domain.e1f.values[0, tran_y, tran_x]
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
        # indices along the transect        
        self.y_idx = tran_y
        self.x_idx = tran_x
        self.len = len(tran_y)
        self.flux_across_AB = None # (time,depth_index,transect_segment_index)
        self.vol_flux_across_AB = None # (time,transect_segment_index)
        # dataset along the transect
        da_tran_y = xa.DataArray( tran_y, dims=['a'])
        da_tran_x = xa.DataArray( tran_x, dims=['a'])
        self.data = dataset.isel(y=da_tran_y, x=da_tran_x)
        # For calculations we need access to a halo of points around the transect
        # self.data_n = dataset.isel(y=tran_y+1,x=tran_x)  
        # self.data_e = dataset.isel(y=tran_y,x=tran_x+1) 
        # self.data_s = dataset.isel(y=tran_y-1,x=tran_x) 
        # self.data_w = dataset.isel(y=tran_y,x=tran_x-1) 
        self.domain = domain.dataset 
        # For calculations we need access to a halo of points around the transect
        # self.domain_n = domain.dataset.isel(y=tran_y+1,x=tran_x)  
        # self.domain_e = domain.dataset.isel(y=tran_y,x=tran_x+1) 
        # self.domain_s = domain.dataset.isel(y=tran_y-1,x=tran_x) 
        # self.domain_w = domain.dataset.isel(y=tran_y,x=tran_x-1) 
        
    def transport_across_AB(self):
        
        flux = []
        vol_flux = []
        dy = np.diff(self.y_idx)
        dx = np.diff(self.x_idx)
        # Loop through each point along the transact
        for idx in np.arange(0, self.len):
            
            if dy[idx] > 0:
                # u flux (+ in)
                u_flux = self.data.uoce.values[:,:,idx+1] * self.domain.e2u.values[:,:,idx+1]
                flux.append( u_flux )
                vol_flux.append( np.sum( u_flux * self.data.e3u.values[:,:,idx+1], axis=1 ) )
            elif dx[idx] > 0:
                # v flux (- in)
                v_flux = -self.data.voce.values[:,:,idx+1] * self.domain.e1v.values[:,:,idx+1]
                flux.append( v_flux )
                vol_flux.append( np.sum( v_flux * self.data.e3v.values[:,:,idx+1], axis=1 ) )             
            elif dx[idx] < 0:
                # v flux (+ in)
                v_flux = self.data.voce.values[:,:,idx] * self.domain.e1v.values[:,:,idx]
                flux.append( v_flux )
                vol_flux.append( np.sum( v_flux * self.data.e3v.values[:,:,idx], axis=1 ) )
        
        self.flux_across_AB = np.asarray(flux)
        self.vol_flux_across_AB = np.asarray(vol_flux)
        return ( np.sum(self.flux_across_AB, axis=2), np.sum(self.vol_flux_across_AB, axis=1) )  
    
