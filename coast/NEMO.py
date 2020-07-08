from .COAsT import COAsT
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):
    
    def __init__(self, fn_data=None, fn_domain=None, grid_ref='t-grid',
                 chunks: dict=None, multiple=False,
                 workers=2, threads=2, memory_limit_per_worker='2GB'):
        self.dataset = xr.Dataset()
        self.grid_ref = grid_ref.lower()
        self.domain_loaded = False
        
        self.set_dimension_mapping()
        self.set_variable_mapping()
        if fn_data is not None:
            self.load(fn_data, chunks, multiple)
        self.set_dimension_names(self.dim_mapping)
        self.set_variable_names(self.var_mapping)
        
        if fn_domain is None:
            print("No NEMO domain specified, only limited functionality"+ 
                  " will be available")
        else:
            dataset_domain = self.load_domain(fn_domain, chunks)
            self.set_timezero_depths(dataset_domain)
            self.merge_domain_into_dataset(dataset_domain)
            
    def set_dimension_mapping(self):
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim',
                            'y':'y_dim', 'x':'x_dim'}
        self.dim_mapping_domain = {'t':'t_dim0', 'x':'x_dim', 'y':'y_dim',
                                   'z':'z_dim'}

    def set_variable_mapping(self):
        self.var_mapping = {'time_counter':'time',
                            'votemper' : 'temperature',
                            'temp' : 'temperature'}
        # NAMES NOT SET IN STONE.
        self.var_mapping_domain = {'time_counter' : 'time0', 
                                   'glamt':'longitude', 'glamu':'longitude', 
                                   'glamv':'longitude','glamf':'longitude',
                                   'gphit':'latitude', 'gphiu':'latitude', 
                                   'gphiv':'latitude', 'gphif':'latitude',
                                   'e1t':'e1', 'e1u':'e1', 
                                   'e1v':'e1', 'e1f':'e1',
                                   'e2t':'e2', 'e2u':'e2', 
                                   'e2v':'e2', 'e2f':'e2',
                                   'ff_t':'ff', 'ff_f':'ff',
                                   'e3t_0':'e3_0', 'e3u_0':'e3_0',
                                   'e3v_0':'e3_0', 'e3f_0':'e3_0',
                                   'deptht_0':'depth_0', 'depthw_0':'depth_0'}

    def load_domain(self, fn_domain, chunks):
        ''' Loads domain file and renames dimensions with dim_mapping_domain'''
        # Load xarrat dataset
        dataset_domain = xr.open_dataset(fn_domain)
        self.domain_loaded = True
        # Rename dimensions
        for key, value in self.dim_mapping_domain.items():
            try:
                dataset_domain = dataset_domain.rename_dims({ key : value })
            except:
                print('pass: {}: {}', key, value)
                pass

        return dataset_domain
   
    def merge_domain_into_dataset(self, dataset_domain):
        ''' Merge domain dataset variables into self.dataset, using grid_ref'''
        
        # Define grid independent variables to pull across
        not_grid_vars = ['jpiglo', 'jpjglo','jpkglo','jperio',
                         'ln_zco', 'ln_zps', 'ln_sco', 'ln_isfcav']
        
        # Define grid specific variables to pull across
        if self.grid_ref == 'u-grid': 
            grid_vars = ['glamu', 'gphiu', 'e1u', 'e2u', 'e3u_0'] #What about e3vw
        elif self.grid_ref == 'v-grid': 
            grid_vars = ['glamv', 'gphiv', 'e1v', 'e2v', 'e3v_0']
        elif self.grid_ref == 't-grid': 
            grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3t_0']
        elif self.grid_ref == 'w-grid': 
            grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3w_0']
        elif self.grid_ref == 'f-grid': 
            grid_vars = ['glamf', 'gphif', 'e1f', 'e2f', 'e3f_0']  
            
        all_vars = grid_vars + not_grid_vars
            
        for var in all_vars:
            try:
                new_name = self.var_mapping_domain[var]
                self.dataset[new_name] = dataset_domain[var].squeeze()
            except:
                pass
            
        # Reset & set specified coordinates
        coord_vars = ['longitude', 'latitude', 'time', 'depth_0']
        self.dataset = self.dataset.reset_coords()
        for var in coord_vars:
            try:
                self.dataset = self.dataset.set_coords(var)
            except:
                pass
        
        # Delete specified variables
        delete_vars = ['nav_lat', 'nav_lon', 'deptht']
        for var in delete_vars:
            try:
                self.dataset = self.dataset.drop(var)
            except:
                pass
        

    def __getitem__(self, name: str):
        return self.dataset[name]

    def set_grid_ref_attr(self):
        self.grid_ref_attr_mapping = {'temperature' : 't-grid',
                                'coast_name_for_u_velocity' : 'u-grid',
                                'coast_name_for_v_velocity' : 'v-grid',
                                'coast_name_for_w_velocity' : 'w-grid',
                                'coast_name_for_vorticity'  : 'f-grid' }
        #self.grid_ref_attr_mapping = None

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller

    
    def set_timezero_depths(self, dataset_domain):
        """
        Calculates the depths at time zero (from the domain_cfg input file) 
        for the appropriate grid.
        
        The depths are assigned to NEMO.dataset.depth_0

        """
        try:
            if self.grid_ref == 't-grid':    
                e3w_0 = np.squeeze( dataset_domain.e3w_0.values )
                depth_0 = np.zeros_like( e3w_0 )  
                depth_0[0,:,:] = 0.5 * e3w_0[0,:,:]    
                depth_0[1:,:,:] = depth_0[0,:,:] + np.cumsum( e3w_0[1:,:,:], axis=0 ) 
            elif self.grid_ref == 'w-grid':            
                e3t_0 = np.squeeze( dataset_domain.e3t_0.values )
                depth_0 = np.zeros_like( e3t_0 ) 
                depth_0[0,:,:] = 0.0
                depth_0[1:,:,:] = np.cumsum( e3t_0, axis=1 )[:-1,:,:]
            elif self.grid_ref == 'u-grid':
                raise ValueError(str(self) + ": u-grid depth calculation not yet implemented")
            elif self.grid_ref == 'v-grid':
                raise ValueError(str(self) + ": v-grid depth calculation not yet implemented")
            elif self.grid_ref == 'f-grid':
                raise ValueError(str(self) + ": f-grid depth calculation not yet implemented")
            
            self.dataset['depth_0'] = xr.DataArray(depth_0,
                    dims=['z_dim', 'y_dim', 'x_dim'],
                    attrs={'units':'m',
                    'standard_name': 'depth at time zero on the {}'.format(self.grid_ref)})
        except ValueError as err:
            print(err)

        return
