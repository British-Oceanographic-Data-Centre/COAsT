from .COAsT import COAsT
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):
    """
    Words to describe the NEMO class
    
    kwargs -- define addition keyworded arguemts for domain file. E.g. ln_sco=1
    if using s-scoord in an old domain file that does not carry this flag.
    """
    def __init__(self, fn_data, fn_domain=None, grid_ref='t-grid',
                 chunks: dict=None, multiple=False,
                 workers=2, threads=2, memory_limit_per_worker='2GB', **kwargs):
        self.dataset = None
        self.grid_ref = grid_ref.lower()
        self.domain_loaded = False
        
        self.set_dimension_mapping()
        self.set_variable_mapping()
        self.load(fn_data, chunks, multiple)
        self.set_dimension_names(self.dim_mapping)
        self.set_variable_names(self.var_mapping)
                
        if fn_domain is None:
            print("No NEMO domain specified, only limited functionality"+ 
                  " will be available")
        else:
            dataset_domain = self.load_domain(fn_domain, chunks)
            # Define extra domain attributes using kwargs dictionary
            for key,value in kwargs.items():
                dataset_domain[key] = value
            print( dataset_domain['x_dim'].size, dataset_domain['y_dim'].size)    
            dataset_domain = self.trim_domain_size( dataset_domain )
            print( dataset_domain['x_dim'].size, dataset_domain['y_dim'].size)    
            self.construct_depths(dataset_domain)
            self.merge_domain_into_dataset(dataset_domain)
            
    def set_dimension_mapping(self):
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim',
                            'y':'y_dim', 'x':'x_dim'}
        self.dim_mapping_domain = {'t':'t_dim0', 'x':'x_dim', 'y':'y_dim',
                                   'z':'z_dim'}

    def set_variable_mapping(self):
        self.var_mapping = {'time_counter':'time',
                            'votemper' : 'temperature',
                            'thetao' : 'temperature',
                            'temp' : 'temperature',
                            'so' : 'salinity'}
        # NAMES NOT SET IN STONE.
        self.var_mapping_domain = {'time_counter' : 'time0', 
                                   'glamt':'longitude', 'glamu':'longitude', 
                                   'glamv':'longitude','glamf':'longitude',
                                   'gphit':'latitude', 'gphiu':'latitude', 
                                   'gphiv':'latitude', 'gphiv':'latitude',
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
            grid_vars = ['glamu', 'gphiu', 'e1u', 'e2u', 'e3u_0', 'deptht_0'] #What about e3vw
            print('CAUTION: ASSIGNING DEPTHS AT T-POINTS. Use depth coord for plotting only')
        elif self.grid_ref == 'v-grid': 
            grid_vars = ['glamv', 'gphiv', 'e1v', 'e2v', 'e3v_0', 'deptht_0']
            print('CAUTION: ASSIGNING DEPTHS AT T-POINTS. Use depth coord for plotting only')
        elif self.grid_ref == 't-grid': 
            grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3t_0', 'deptht_0']
        elif self.grid_ref == 'w-grid': 
            grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3w_0', 'depthw_0']
        elif self.grid_ref == 'f-grid': 
            grid_vars = ['glamf', 'gphif', 'e1f', 'e2f', 'e3f_0', 'deptht_0']  
            print('CAUTION: ASSIGNING DEPTHS AT T-POINTS. Use depth coord for plotting only')
            
        all_vars = grid_vars + not_grid_vars
        
        # Trim domain DataArray area if necessary. 
        self.copy_domain_vars_to_dataset( dataset_domain, grid_vars )

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

    def construct_depths(self, dataset_domain):
        # Construct depths
        # xarray method for parsing depth variables:  # --> depth_t ,depth_w
        print("In construct_depths")
        if dataset_domain['ln_sco'].values == 1:
            dataset_domain['deptht_0'], dataset_domain['depthw_0'] = \
                self.get_depth_as_xr(dataset_domain.e3t_0, dataset_domain.e3w_0)
        else:
            print('Reconstruct depths for grids with ln_sco = 1. Not tried other grids yet')
        return

    def get_depth(self, e3t: np.ndarray, e3w: np.ndarray=None ):
        """
        Returns the depth at t and w points.
        If the w point scale factors are missing an approximation is made.
        :param e3t: vertical scale factors at t points
        :param e3w: (optional) vertical scale factors at w points.
        :return: tuple of 2 4d arrays (time,z_dim,y_dim,x_dim) containing depth at t
                    and w points respectively
        """

        depth_t = np.ma.empty_like( e3t )
        depth_w = np.ma.empty_like( e3t )
        depth_w[:,0,:,:] = 0.0
        depth_w[:,1:,:,:] = np.cumsum( e3t, axis=1 )[:,:-1,:,:]
        if e3w is not None:
            depth_t[:,0,:,:] = 0.5 * e3w[:,0,:,:]
            depth_t[:,1:,:,:] =  depth_t[:,0,:,:] + np.cumsum( e3w[:,1:,:,:], axis=1 )
        else:
            depth_t[:,:-1,:,:] = 0.5 * ( depth_w[:,:-1,:,:] + depth_w[:,1:,:,:] )
            depth_t[:,-1,:,:] = np.nan

        return (np.ma.masked_invalid(depth_t), np.ma.masked_invalid(depth_w))


    def get_depth_as_xr( self, e3t: xr.DataArray, e3w: xr.DataArray=None ):
        """
        Inputs and outputs as xarray DataArrays
        Returns the depth at t and w points.
        If the w point scale factors are missing an approximation is made.
        :param e3t: vertical scale factors at t points
        :param e3w: (optional) vertical scale factors at w points.
        :return: tuple of 2 4d arrays (t_dim,z_dim,y_dim,x_dim) containing depth at t
                    and w points respectively
                if t_dim has only one value. This dimension is squeezed.
        """
        depth_t = np.ma.empty_like( e3t.values )
        depth_w = np.ma.empty_like( e3t.values )
        depth_w[:,0,:,:] = 0.0
        depth_w[:,1:,:,:] = np.cumsum( e3t.values, axis=1 )[:,:-1,:,:]

        if e3w is not None:
            depth_t[:,0,:,:] = 0.5 * e3w.values[:,0,:,:]
            depth_t[:,1:,:,:] =  depth_t[:,0,:,:] + np.cumsum( e3w.values[:,1:,:,:], axis=1 )
        else:
            depth_t[:,:-1,:,:] = 0.5 * ( depth_w[:,:-1,:,:] + depth_w[:,1:,:,:] )
            depth_t[:,-1,:,:] = np.nan

        depth_t_xr = xr.DataArray( np.ma.masked_invalid(depth_t),
                            dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'],
                            attrs={'grid' : 't',
                                   'units':'m',
                                   'standard_name': 'depth on t-points'}).squeeze()

        depth_w_xr = xr.DataArray( np.ma.masked_invalid(depth_w),
                            dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'],
                            attrs={'grid': 'w',
                                   'units':'m',
                                   'standard_name': 'depth on w-points'}).squeeze()

        return depth_t_xr, depth_w_xr
    
    def trim_domain_size( self, dataset_domain ):
        """
        Trim the domain variables if the dataset object is a spatial subset
        """
        if (self.dataset['x_dim'].size != dataset_domain['x_dim'].size)  \
                or (self.dataset['y_dim'].size != dataset_domain['y_dim'].size):
            #print("The domain and dataset object are different sizes. Trim domain")
            # Find the corners of the cut out domain.
            [j0,i0] = self.find_j_i( self.dataset.nav_lat[0,0], 
                                    self.dataset.nav_lon[0,0], dataset_domain )
            [j1,i1] = self.find_j_i( self.dataset.nav_lat[-1,-1], 
                                    self.dataset.nav_lon[-1,-1], dataset_domain )

            dataset_subdomain = dataset_domain.isel( 
                                        y_dim = slice(j0, j1 + 1),
                                        x_dim = slice(i0, i1 + 1) ) 
            #print (dataset_subdomain)
            return dataset_subdomain
        else:
            return dataset_domain

    def copy_domain_vars_to_dataset( self, dataset_domain, grid_vars ):
        """
        Map the domain coordand metric variables to the dataset object.
        Expects the source and target DataArrays to be same sizes.
        """
        for var in grid_vars:        
            try:
                new_name = self.var_mapping_domain[var]
                self.dataset[new_name] = dataset_domain[var].squeeze()                    
                #print("map: {} --> {}".format( var, new_name))
            except:
                pass
    
    
  
                
    def find_j_i(self, lat, lon, dataset_domain):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12, t)

        :param lat: latitude
        :param lon: longitude
        :param grid_ref: the gphi/glam version a user wishes to search over
        :return: the y and x coordinates for the given grid_ref variable within the domain file
        """

        internal_lat = dataset_domain[f"gphi{self.grid_ref.replace('-grid','')}"]
        internal_lon = dataset_domain[f"glam{self.grid_ref.replace('-grid','')}"]
        dist2 = xr.ufuncs.square(internal_lat - lat) \
                + xr.ufuncs.square(internal_lon - lon)
        [_, y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]