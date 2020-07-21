from .COAsT import COAsT
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt
import gsw
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import warnings
  
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
                                   'deptht_0':'depth_0', 'depthw_0':'depth_0',
                                   'ln_sco':'ln_sco'}

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
    
    
    def construct_density_onto_z( self, EOS='EOS10', z_levels=None ):        
        
        # Not that this is a performance intensive process and data should be subsetted prior
        # to performing it. 
        
        try:
            if EOS != 'EOS10': 
                raise ValueError(str(self) + ': Density calculation for ' + EOS + ' not implemented.')
            if self.grid_ref != 't-grid':
                raise ValueError(str(self) + ': Density calculation can only be performed for a t-grid object,\
                                 the tracer grid for NEMO.' )
            if not self.dataset.ln_sco.item():
                raise ValueError(str(self) + ': Density calculation only implemented for s-vertical-coordinates.')            

            # If caller does not specify a depth profile to regrid onto then
            # use the average depth_0 (across horiztonal space).
            # NOTE: when time dependent depth is coded up we should use that instead
            if z_levels is None:
                z_levels = self.dataset.depth_0.max(dim=(['x_dim','y_dim']))
            
                #z_levels = self.dataset.depth_0.mean( ('x_dim', 'y_dim'), skipna=True )
            
            density = np.ma.zeros( ( self.dataset.t_dim.size, z_levels.size, 
                                    self.dataset.y_dim.size, self.dataset.x_dim.size ) )
            #density = np.ma.zeros( (1,z_levels.size, 1,1) )

            #for it in self.dataset.t_dim:
            for it in np.arange(0,1): 
                #for iy in self.dataset.y_dim:
                for iy in np.arange(172,173):
                    #for ix in self.dataset.x_dim: 
                    for ix in np.arange(154,155):
                        if np.all(xr.ufuncs.isnan(self.dataset.vosaline[it,:,iy,ix]).values):
                            density[it,:,iy,ix] = np.nan
                            density[it,:,iy,ix].mask = True
                        else:
                            sal = self.dataset.vosaline[it,:,iy,ix].to_masked_array()
                            temp = self.dataset.temperature[it,:,iy,ix].to_masked_array()
                            s_levels = self.dataset.depth_0[:,iy,ix].to_masked_array()
                            lat = self.dataset.latitude[iy, ix]
                            lon = self.dataset.longitude[iy, ix]
                        
                            sal_func = interp1d( s_levels[sal.mask==False], sal[sal.mask==False], 
                                        bounds_error=False, kind='linear', fill_value='extrapolate')
                            temp_func = interp1d( s_levels[temp.mask==False], temp[temp.mask==False], 
                                        bounds_error=False, kind='linear', fill_value='extrapolate')
                            
                            pressure_absolute = gsw.p_from_z( -z_levels, lat ) # depth must be negative           
                            print(pressure_absolute)                            
                            sal_absolute = gsw.SA_from_SP( sal_func( z_levels ), pressure_absolute, lon, lat )  
                            # These values will end up being masked but negative values raise warnings.
                            sal_absolute[sal_absolute < 0]=0 
                            print(temp_func( z_levels ))
                            print(sal_absolute)
                            print(s_levels)
                            
                            temp_conservative = gsw.CT_from_pt( sal_absolute, temp_func( z_levels ) )

                            # with warnings.catch_warnings():
                            #     warnings.filterwarnings('error')
                            #     try:
                            #         pressure_absolute = gsw.p_from_z( -z_levels, lat ) # depth must be negative           
                            #         sal_absolute = gsw.SA_from_SP( sal_func( z_levels ), pressure_absolute, lon, lat )   
                            #         temp_conservative = gsw.CT_from_pt( sal_absolute, temp_func( z_levels ) )
                            #     except Warning as e:
                            #         print('ix: ' + str(ix) + ' iy: ' + str(iy))
                            
                            s_level_bottom = s_levels[s_levels.mask==False][-1]
                            sal_absolute[z_levels > s_level_bottom] = np.nan
                            temp_conservative[z_levels > s_level_bottom] = np.nan
                            density[it,:,iy,ix] = np.ma.masked_invalid( gsw.rho( 
                                sal_absolute, temp_conservative, pressure_absolute ), np.nan )

            self.dataset['density_z_levels'] = xr.DataArray( density, 
                    coords={'time': (('t_dim'), self.dataset.time.values),
                            'depth_z_levels': (('z_dim'), z_levels.values),
                            'latitude': (('y_dim','x_dim'), self.dataset.latitude.values),
                            'longitude': (('y_dim','x_dim'), self.dataset.longitude.values)},
                    dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'] )


        except ValueError as err:
            print(err)
            
        return
    
    def construct_density_on_z_levels( self, EOS='EOS10', z_levels=None ):        
        
        # Not that this is a very time consuming process and data should be subsetted or averaged
        # in space and or time prior to performing it rather than after. 
        
        try:
            if EOS != 'EOS10': 
                raise ValueError(str(self) + ': Density calculation for ' + EOS + ' not implemented.')
            if self.grid_ref != 't-grid':
                raise ValueError(str(self) + ': Density calculation can only be performed for a t-grid object,\
                                 the tracer grid for NEMO.' )
            if not self.dataset.ln_sco.item():
                raise ValueError(str(self) + ': Density calculation only implemented for s-vertical-coordinates.')            
    
            if z_levels is None:
                z_levels = self.dataset.depth_0.max(dim=(['x_dim','y_dim']))                
                z_levels_min = self.dataset.depth_0[0,:,:].max(dim=(['x_dim','y_dim']))
                z_levels[0] = z_levels_min
            
            sal_z_levels = np.ma.zeros( ( self.dataset.t_dim.size, z_levels.size, 
                            self.dataset.y_dim.size, self.dataset.x_dim.size ) )
            temp_z_levels = np.ma.zeros( ( self.dataset.t_dim.size, z_levels.size, 
                            self.dataset.y_dim.size, self.dataset.x_dim.size ) )
            density_z_levels = np.ma.zeros( ( self.dataset.t_dim.size, z_levels.size, 
                            self.dataset.y_dim.size, self.dataset.x_dim.size ) )
            
            sal = self.dataset.vosaline.to_masked_array()
            temp = self.dataset.temperature.to_masked_array()
            s_levels = self.dataset.depth_0.to_masked_array()
            lat = self.dataset.latitude.values
            lon = self.dataset.longitude.values
    
            for it in self.dataset.t_dim:
            #for it in np.arange(0,1): 
                for iy in self.dataset.y_dim:
                #for iy in np.arange(172,173):
                    for ix in self.dataset.x_dim: 
                    #for ix in np.arange(154,155):
                        if np.all(xr.ufuncs.isnan(self.dataset.vosaline[it,:,iy,ix]).values):
                            density_z_levels[it,:,iy,ix] = np.nan
                            density_z_levels[it,:,iy,ix].mask = True
                        else:                      
                            sal_func = interp1d( s_levels[:,iy,ix], sal[it,:,iy,ix], 
                                        bounds_error=False, kind='linear')
                            temp_func = interp1d( s_levels[:,iy,ix], temp[it,:,iy,ix], 
                                        bounds_error=False, kind='linear')
                            
                            sal_z_levels[it,:,iy,ix] = sal_func(z_levels.values)
                            temp_z_levels[it,:,iy,ix] = temp_func(z_levels.values)
                
            
            pressure_absolute = np.ma.masked_invalid(
                gsw.p_from_z( -z_levels.values[:,np.newaxis,np.newaxis], lat ) ) # depth must be negative           
                       
            sal_absolute = gsw.SA_from_SP( sal_z_levels, pressure_absolute, lon, lat )  
            # These values will end up being masked but negative values raise warnings.
            sal_absolute[sal_absolute < 0]=np.nan
            sal_absolute = np.ma.masked_invalid(sal_absolute)
            temp_conservative = np.ma.masked_invalid(
                gsw.CT_from_pt( sal_absolute, temp_z_levels ) )
    
            density_z_levels = np.ma.masked_invalid( gsw.rho( 
                sal_absolute, temp_conservative, pressure_absolute ), np.nan )
    
            self.dataset['density_z_levels'] = xr.DataArray( density_z_levels, 
                    coords={'time': (('t_dim'), self.dataset.time.values),
                            'depth_z_levels': (('z_dim'), z_levels.values),
                            'latitude': (('y_dim','x_dim'), self.dataset.latitude.values),
                            'longitude': (('y_dim','x_dim'), self.dataset.longitude.values)},
                    dims=['t_dim', 'z_dim', 'y_dim', 'x_dim'] )


        except ValueError as err:
            print(err)
            
        return

 