from .COAsT import COAsT
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt
import sklearn.neighbors as nb

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
            pass
            #print("No NEMO domain specified, only limited functionality"+ 
            #      " will be available")
        else:
            dataset_domain = self.load_domain(fn_domain, chunks)
            self.set_timezero_depths(dataset_domain)
            self.merge_domain_into_dataset(dataset_domain)
            
    def set_dimension_mapping(self):
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim', 
                            'depthu':'z_dim', 'depthv':'z_dim',
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
                depth_0[1:,:,:] = np.cumsum( e3t_0, axis=0 )[:-1,:,:]
            elif self.grid_ref == 'u-grid':
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_u = 0.5 * ( e3w_0[:,:,:-1] + e3w_0[:,:,1:] )
                depth_0 = np.zeros_like( e3w_0 )  
                depth_0[0,:,:-1] = 0.5 * e3w_0_on_u[0,:,:]    
                depth_0[1:,:,:-1] = depth_0[0,:,:-1] + np.cumsum( e3w_0_on_u[1:,:,:], axis=0 ) 
            elif self.grid_ref == 'v-grid':
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_v = 0.5 * ( e3w_0[:,:-1,:] + e3w_0[:,1:,:] )
                depth_0 = np.zeros_like( e3w_0 )  
                depth_0[0,:-1,:] = 0.5 * e3w_0_on_v[0,:,:]    
                depth_0[1:,:-1,:] = depth_0[0,:-1,:] + np.cumsum( e3w_0_on_v[1:,:,:], axis=0 ) 
            elif self.grid_ref == 'f-grid':
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_f = 0.25 * ( e3w_0[:,:-1,:-1] + e3w_0[:,1:,:-1] +
                                     e3w_0[:,:-1,:-1] + e3w_0[:,:-1,1:] )
                depth_0 = np.zeros_like( e3w_0 ) 
                depth_0[0,:-1,:-1] = 0.5 * e3w_0_on_f[0,:,:]    
                depth_0[1:,:-1,:-1] = depth_0[0,:-1,:-1] + np.cumsum( e3w_0_on_f[1:,:,:], axis=0 ) 
            else:
                raise ValueError(str(self) + ": " + self.grid_ref + " depth calculation not implemented")

            self.dataset['depth_0'] = xr.DataArray(depth_0,
                    dims=['z_dim', 'y_dim', 'x_dim'],
                    attrs={'Units':'m',
                    'standard_name': 'Depth at time zero on the {}'.format(self.grid_ref)})
        except ValueError as err:
            print(err)

        return
    
    
    def find_j_i(self, lat: int, lon: int):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12)

        :param lat: latitude
        :param lon: longitude
        :return: the y and x coordinates for the NEMO object's grid_ref, i.e. t,u,v,f,w.
        """

        dist2 = xr.ufuncs.square(self.dataset.latitude - lat) + xr.ufuncs.square(self.dataset.longitude - lon)
        [y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    
    def transect_indices(self, start: tuple, end: tuple) -> tuple:
        """
        This method returns the indices of a simple straight line transect between two 
        lat lon points defined on the NEMO object's grid_ref, i.e. t,u,v,f,w.

        :type start: tuple A lat/lon pair
        :type end: tuple A lat/lon pair
        :return: array of y indices, array of x indices, number of indices in transect
        """

        [j1, i1] = self.find_j_i(start[0], start[1])  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1])  # lat , lon

        line_length = max(np.abs(j2 - j1), np.abs(i2 - i1)) + 1

        jj1 = [int(jj) for jj in np.round(np.linspace(j1, j2, num=line_length))]
        ii1 = [int(ii) for ii in np.round(np.linspace(i1, i2, num=line_length))]
        
        return jj1, ii1, line_length
    
    def nearest_xy_indices(self, model_dataset, new_lons, new_lats):
        '''
        Obtains the x and y indices of the nearest model points to specified
        lists of longitudes and latitudes. Makes use of sklearn.neighbours
        and its BallTree haversine method. 
        
        Example Useage
        ----------
        # Get indices of model points closest to altimetry points
        ind_x, ind_y = nemo.nearest_indices(altimetry.dataset.longitude,
                                            altimetry.dataset.latitude)
        # Nearest neighbour interpolation of model dataset to these points
        interpolated = nemo.dataset.isel(x_dim = ind_x, y_dim = ind_y)

        Parameters
        ----------
        model_dataset (xr.Dataset or xr.DataArray): model xarray dataset.
            Must contain coordinates.
        new_lons (array): Array of longitudes (degrees) to compare with model
        new_lats (array): Array of latitudes (degrees) to compare with model
        
        Returns
        -------
        Array of x indices, Array of y indices
        '''
        # Cast lat/lon to numpy arrays
        new_lons = np.array(new_lons)
        new_lats = np.array(new_lats)
        mod_lon = np.array(model_dataset.longitude).flatten()
        mod_lat = np.array(model_dataset.latitude).flatten()
        
        # Put lons and lats into 2D location arrays for BallTree: [lat, lon]
        mod_locs = np.vstack((mod_lat, mod_lon)).transpose()
        new_locs = np.vstack((new_lats, new_lons)).transpose()
        
        # Convert lat/lon to radians for BallTree
        mod_locs = np.radians(mod_locs)
        new_locs = np.radians(new_locs)
        
        # Do nearest neighbour interpolation using BallTree (gets indices)
        tree = nb.BallTree(mod_locs, leaf_size=5, metric='haversine')
        _, ind_1d = tree.query(new_locs, k=1)
        
        # Get 2D indices from 1D index output from BallTree
        ind_y, ind_x = np.unravel_index(ind_1d, model_dataset.longitude.shape)
        ind_x = xr.DataArray(ind_x.squeeze())
        ind_y = xr.DataArray(ind_y.squeeze())

        return ind_x, ind_y
    
    def nearest_time_indices(self):
        raise NotImplementedError
        return
    
    def nearest_depth_indices(self):
        raise NotImplementedError
        return
    
    def interpolate_in_space(self, model_array, new_lons, new_lats):
        '''
        Interpolates a provided xarray.DataArray in space to new longitudes
        and latitudes using a nearest neighbour method.
        
        Example Useage
        ----------

        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_lons (array): Array of longitudes (degrees) to compare with model
        new_lats (array): Array of latitudes (degrees) to compare with model
        
        Returns
        -------
        Interpolated DataArray
        '''
        
        # Get nearest indices
        ind_x, ind_y = self.nearest_xy_indices(model_array, new_lons, new_lats)
        
        # Geographical interpolation (using BallTree indices)
        interpolated = model_array.isel(x_dim=ind_x, y_dim=ind_y)
        interpolated = interpolated.rename({'dim_0':'interp_dim'})
        return interpolated
    
    def interpolate_in_time(self, model_array, new_times, 
                               interp_method = 'nearest', extrapolate=True):
        '''
        Interpolates a provided xarray.DataArray in space to new longitudes
        and latitudes using a specified scipy.interpolate method.
        
        Example Useage
        ----------

        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_times (array): New times to interpolate to (array of datetimes)
        interp_method (str): Interpolation method
        
        Returns
        -------
        Interpolated DataArray
        '''
        
        # Time interpolation
        interpolated = model_array.swap_dims({'t_dim':'time'})
        if extrapolate:
            interpolated = interpolated.interp(time = new_times,
                                           method = interp_method,
                                           kwargs={'fill_value':'extrapolate'})
        else:
            interpolated = interpolated.interp(time = new_times,
                                           method = interp_method)
        #interpolated = interpolated.swap_dims({'time':'t_dim'})
        
        return interpolated
    
    def interpolate_in_depth(self, model_array, new_depths):
        raise NotImplementedError
        return
