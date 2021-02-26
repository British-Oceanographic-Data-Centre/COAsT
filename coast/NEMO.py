from .COAsT import COAsT
from . import general_utils, stats_util
import xarray as xr
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import gsw
import warnings
from .logging_util import get_slug, debug, info, warn, error



class NEMO(COAsT):  # TODO Complete this docstring
    """
    Words to describe the NEMO class

    kwargs -- define addition keyworded arguemts for domain file. E.g. ln_sco=1
    if using s-scoord in an old domain file that does not carry this flag.
    """
    def __init__(self, fn_data=None, fn_domain=None, grid_ref='t-grid',  # TODO Super init not called + add a docstring
                 chunks: dict=None, multiple=False,
                 workers=2, threads=2, memory_limit_per_worker='2GB', **kwargs):
        debug(f"Creating new {get_slug(self)}")
        self.dataset = xr.Dataset()
        self.grid_ref = grid_ref.lower()
        self.domain_loaded = False

        self.set_grid_vars()
        self.set_dimension_mapping()
        self.set_variable_mapping()
        if fn_data is not None:
            self.load(fn_data, chunks, multiple)
        self.set_dimension_names(self.dim_mapping)
        self.set_variable_names(self.var_mapping)

        if fn_domain is None:
            self.filename_domain = "" # empty store for domain fileanme
            warn("No NEMO domain specified, only limited functionality"+
                 " will be available")
        else:
            self.filename_domain = fn_domain # store domain fileanme
            dataset_domain = self.load_domain(fn_domain, chunks)

            # Define extra domain attributes using kwargs dictionary
            ## This is a bit of a placeholder. Some domain/nemo files will have missing variables
            for key,value in kwargs.items():
                dataset_domain[key] = value

            if fn_data is not None:
                dataset_domain = self.trim_domain_size( dataset_domain )
            self.set_timezero_depths(dataset_domain) # THIS ADDS TO dataset_domain. Should it be 'return'ed (as in trim_domain_size) or is implicit OK?
            self.merge_domain_into_dataset(dataset_domain)
            debug(f"Initialised {get_slug(self)}")

    def set_grid_vars(self):
        """ Define the variables to map from the domain file to the NEMO obj"""
        # Define grid specific variables to pull across
        if self.grid_ref == 'u-grid':
            self.grid_vars = ['glamu', 'gphiu', 'e1u', 'e2u', 'e3u_0', 'depthu_0'] #What about e3vw
        elif self.grid_ref == 'v-grid':
            self.grid_vars = ['glamv', 'gphiv', 'e1v', 'e2v', 'e3v_0', 'depthv_0']
        elif self.grid_ref == 't-grid':
            self.grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3t_0', 'deptht_0', 'tmask']
        elif self.grid_ref == 'w-grid':
            self.grid_vars = ['glamt', 'gphit', 'e1t', 'e2t', 'e3w_0', 'depthw_0']
        elif self.grid_ref == 'f-grid':
            self.grid_vars = ['glamf', 'gphif', 'e1f', 'e2f', 'e3f_0', 'depthf_0']


    def set_dimension_mapping(self):  # TODO Add a docstring
        self.dim_mapping = {'time_counter':'t_dim', 'deptht':'z_dim',
                            'depthu':'z_dim', 'depthv':'z_dim',
                            'y':'y_dim', 'x':'x_dim',
                            'x_grid_T':'x_dim', 'y_grid_T':'y_dim'}
        debug(f"{get_slug(self)} dim_mapping set to {self.dim_mapping}")
        self.dim_mapping_domain = {'t':'t_dim0', 'x':'x_dim', 'y':'y_dim',
                                   'z':'z_dim'}
        debug(f"{get_slug(self)} dim_mapping_domain set to {self.dim_mapping_domain}")

    def set_variable_mapping(self):  # TODO Add a docstring
        # Variable names remapped  within NEMO object
        self.var_mapping = {'time_counter':'time',
                            'votemper' : 'temperature',
                            'thetao' : 'temperature',
                            'temp' : 'temperature',
                            'toce' : 'temperature',
                            'so' : 'salinity',
                            'vosaline' : 'salinity',
                            'voce' : 'salinity',
                            'sossheig' : 'ssh',
                            'zos' : 'ssh' }
        # Variable names mapped from domain to NEMO object
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
                                   'e3t_0':'e3_0', 'e3w_0':'e3_0',
                                   'e3u_0':'e3_0', 'e3v_0':'e3_0',
                                   'e3f_0':'e3_0',
                                   'tmask':'mask',
                                   'depthf_0':'depth_0',
                                   'depthu_0':'depth_0', 'depthv_0':'depth_0',
                                   'depthw_0':'depth_0', 'deptht_0':'depth_0',
                                   'ln_sco':'ln_sco'}

    # TODO Add parameter type hints and a docstring
    def load_domain(self, fn_domain, chunks):  # TODO Do something with this unused parameter or remove it
        ''' Loads domain file and renames dimensions with dim_mapping_domain'''
        # Load xarray dataset
        info(f"Loading domain: \"{fn_domain}\"")
        dataset_domain = xr.open_dataset(fn_domain)
        self.domain_loaded = True
        # Rename dimensions
        for key, value in self.dim_mapping_domain.items():
            mapping = {key: value}
            try:
                dataset_domain = dataset_domain.rename_dims(mapping)
            except:  # FIXME Catch specific exception(s)
                error(f"Exception while renaming dimensions from domain in NEMO object with key:value {mapping}")

        return dataset_domain

    def merge_domain_into_dataset(self, dataset_domain):
        ''' Merge domain dataset variables into self.dataset, using grid_ref'''
        debug(f"Merging {get_slug(dataset_domain)} into {get_slug(self)}")
        # Define grid independent variables to pull across
        not_grid_vars = ['jpiglo', 'jpjglo','jpkglo','jperio',
                         'ln_zco', 'ln_zps', 'ln_sco', 'ln_isfcav']

        all_vars = self.grid_vars + not_grid_vars  # FIXME Add an else clause to avoid unhandled error when no ifs are True

        # Trim domain DataArray area if necessary.
        self.copy_domain_vars_to_dataset( dataset_domain, self.grid_vars )

        # Reset & set specified coordinates
        coord_vars = ['longitude', 'latitude', 'time', 'depth_0']
        self.dataset = self.dataset.reset_coords()
        for var in coord_vars:
            try:
                self.dataset = self.dataset.set_coords(var)
            except:  # FIXME Catch specific exception(s)
                pass  # TODO Do we need to log something here?

        # Delete specified variables
        # TODO MIGHT NEED TO DELETE OTHER DEPTH VARS ON OTHER GRIDS?
        delete_vars = ['nav_lat', 'nav_lon', 'deptht']
        for var in delete_vars:
            try:
                self.dataset = self.dataset.drop(var)
            except:  # FIXME Catch specific exception(s)
                pass  # TODO Do we need to log something here?

    def __getitem__(self, name: str):
        return self.dataset[name]

    def set_grid_ref_attr(self):
        debug(f"{get_slug(self)} grid_ref_attr set to {self.grid_ref_attr_mapping}")
        self.grid_ref_attr_mapping = {'temperature' : 't-grid',
                                'coast_name_for_u_velocity' : 'u-grid',
                                'coast_name_for_v_velocity' : 'v-grid',
                                'coast_name_for_w_velocity' : 'w-grid',
                                'coast_name_for_vorticity'  : 'f-grid' }

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        debug(f"Fetching contour complex from {get_slug(self)}")
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller

    def set_timezero_depths(self, dataset_domain):
        """
        Calculates the depths at time zero (from the domain_cfg input file)
        for the appropriate grid.
        The depths are assigned to domain_dataset.depth_0
        """
        debug(f"Setting timezero depths for {get_slug(self)} with {get_slug(dataset_domain)}")
        
        try:
            bathymetry = dataset_domain.bathy_metry.squeeze()
        except AttributeError:
            bathymetry = xr.zeros_like(dataset_domain.e1t.squeeze())
            (warnings.warn(f"The model domain loaded, '{self.filename_domain}', does not contain the "
                          "bathy_metry' variable. This will result in the "
                          "NEMO.dataset.bathymetry variable being set to zero, which "
                          "may result in unexpected behaviour from routines that require "
                          "this variable."))
            debug(f"The bathy_metry variable was missing from the domain_cfg for "
                  f"{get_slug(self)} with {get_slug(dataset_domain)}")
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
                bathymetry[:,:-1] = 0.5 * ( bathymetry[:,:-1] + bathymetry[:,1:] )  
            elif self.grid_ref == 'v-grid':
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_v = 0.5 * ( e3w_0[:,:-1,:] + e3w_0[:,1:,:] )
                depth_0 = np.zeros_like( e3w_0 )
                depth_0[0,:-1,:] = 0.5 * e3w_0_on_v[0,:,:]
                depth_0[1:,:-1,:] = depth_0[0,:-1,:] + np.cumsum( e3w_0_on_v[1:,:,:], axis=0 )
                bathymetry[:-1,:] = 0.5 * ( bathymetry[:-1,:] + bathymetry[1:,:] )   
            elif self.grid_ref == 'f-grid':
                e3w_0 = dataset_domain.e3w_0.values.squeeze()
                e3w_0_on_f = 0.25 * ( e3w_0[:,:-1,:-1] + e3w_0[:,:-1,1:] +
                                     e3w_0[:,1:,:-1] + e3w_0[:,1:,1:] )
                depth_0 = np.zeros_like( e3w_0 )
                depth_0[0,:-1,:-1] = 0.5 * e3w_0_on_f[0,:,:]
                depth_0[1:,:-1,:-1] = depth_0[0,:-1,:-1] + np.cumsum( e3w_0_on_f[1:,:,:], axis=0 )
                bathymetry[:-1,:-1] = 0.25 * ( bathymetry[:-1,:-1] + bathymetry[:-1,1:] 
                                             + bathymetry[1:,:-1] + bathymetry[1:,1:] )  
            else:
                raise ValueError(str(self) + ": " + self.grid_ref + " depth calculation not implemented")
            # Write the depth_0 variable to the domain_dataset DataSet, with grid type
            dataset_domain[f"depth{self.grid_ref.replace('-grid','')}_0"] = xr.DataArray(depth_0,
                    dims=['z_dim', 'y_dim', 'x_dim'],
                    attrs={'units':'m',
                    'standard_name': 'Depth at time zero on the {}'.format(self.grid_ref)})
            self.dataset['bathymetry'] = bathymetry
            self.dataset['bathymetry'].attrs = {'units': 'm','standard_name':'bathymetry',
                'description':'depth of last wet w-level on the horizontal {}'.format(self.grid_ref)}
        except ValueError as err:
            error(err)

    # Add subset method to NEMO class
    def subset_indices(self, start: tuple, end: tuple) -> tuple:
        """
        based on transect_indices, this method looks to return all indices between the given points.
        This results in a 'box' (Quadrilateral) of indices.
        consequently the returned lists may have different lengths.
        :param start: A lat/lon pair
        :param end: A lat/lon pair
        :return: list of y indices, list of x indices,
        """
        debug(f"Subsetting {get_slug(self)} indices from {start} to {end}")
        [j1, i1] = self.find_j_i(start[0], start[1])  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1])  # lat , lon

        return list(np.arange(j1, j2+1)), list(np.arange(i1, i2+1))

    def find_j_i(self, lat: float, lon: float):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12)

        :param lat: latitude
        :param lon: longitude
        :return: the y and x coordinates for the NEMO object's grid_ref, i.e. t,u,v,f,w.
        """
        debug(f"Finding j,i for {lat},{lon} from {get_slug(self)}")
        dist2 = xr.ufuncs.square(self.dataset.latitude - lat) + xr.ufuncs.square(self.dataset.longitude - lon)
        [y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def find_j_i_domain(self, lat: float, lon: float, dataset_domain: xr.DataArray):
        # TODO add dataset_domain to docstring and remove nonexistent grid_ref
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12, dataset_domain)

        :param lat: latitude
        :param lon: longitude
        :param grid_ref: the gphi/glam version a user wishes to search over
        :return: the y and x coordinates for the given grid_ref variable within the domain file
        """
        debug(f"Finding j,i domain for {lat},{lon} from {get_slug(self)} using {get_slug(dataset_domain)}")
        internal_lat = dataset_domain[self.grid_vars[1]] #[f"gphi{self.grid_ref.replace('-grid','')}"]
        internal_lon = dataset_domain[self.grid_vars[0]] #[f"glam{self.grid_ref.replace('-grid','')}"]
        dist2 = xr.ufuncs.square(internal_lat - lat) \
              + xr.ufuncs.square(internal_lon - lon)
        [_, y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def transect_indices(self, start: tuple, end: tuple) -> tuple:
        """
        This method returns the indices of a simple straight line transect between two
        lat lon points defined on the NEMO object's grid_ref, i.e. t,u,v,f,w.

        :type start: tuple A lat/lon pair
        :type end: tuple A lat/lon pair
        :return: array of y indices, array of x indices, number of indices in transect
        """
        debug(f"Fetching transect indices for {start} to {end} from {get_slug(self)}")
        [j1, i1] = self.find_j_i(start[0], start[1])  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1])  # lat , lon

        line_length = max(np.abs(j2 - j1), np.abs(i2 - i1)) + 1

        jj1 = [int(jj) for jj in np.round(np.linspace(j1, j2, num=line_length))]
        ii1 = [int(ii) for ii in np.round(np.linspace(i1, i2, num=line_length))]

        return jj1, ii1, line_length
    
    @staticmethod
    def interpolate_in_space(model_array, new_lon, new_lat, mask=None):
        '''
        Interpolates a provided xarray.DataArray in space to new longitudes
        and latitudes using a nearest neighbour method (BallTree).
        
        Example Usage
        ----------
        # Get an interpolated DataArray for temperature onto two locations
        interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                                 [0,1], [45,46])
        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_lons (1Darray): Array of longitudes (degrees) to compare with model
        new_lats (1Darray): Array of latitudes (degrees) to compare with model
        mask (2D array): Mask array. Where True (or 1), elements of array will
                     not be included. For example, use to mask out land in 
                     case it ends up as the nearest point.
        
        Returns
        -------
        Interpolated DataArray
        '''
        debug(f"Interpolating {get_slug(model_array)} in space with nearest neighbour")
        # Get nearest indices
        ind_x, ind_y = general_utils.nearest_indices_2D(model_array.longitude,
                                                model_array.latitude,
                                                new_lon, new_lat, mask=mask)
        
        # Geographical interpolation (using BallTree indices)
        interpolated = model_array.isel(x_dim=ind_x, y_dim=ind_y)
        if 'dim_0' in interpolated.dims:
            interpolated = interpolated.rename({'dim_0':'interp_dim'})
        return interpolated
    
    @staticmethod
    def interpolate_in_time(model_array, new_times, 
                               interp_method = 'nearest', extrapolate=True):
        '''
        Interpolates a provided xarray.DataArray in time to new python
        datetimes using a specified scipy.interpolate method.
        
        Example Useage
        ----------
        # Get an interpolated DataArray for temperature onto altimetry times
        new_times = altimetry.dataset.time
        interpolated = nemo.interpolate_in_space(nemo.dataset.votemper,
                                                 new_times)
        Parameters
        ----------
        model_array (xr.DataArray): Model variable DataArray to interpolate
        new_times (array): New times to interpolate to (array of datetimes)
        interp_method (str): Interpolation method
        
        Returns
        -------
        Interpolated DataArray
        '''
        debug(f"Interpolating {get_slug(model_array)} in time with method \"{interp_method}\"")
        # Time interpolation
        interpolated = model_array.swap_dims({'t_dim':'time'})
        if extrapolate:
            interpolated = interpolated.interp(time = new_times,
                                           method = interp_method,
                                           kwargs={'fill_value':'extrapolate'})
        else:
            interpolated = interpolated.interp(time = new_times,
                                           method = interp_method)
        # interpolated = interpolated.swap_dims({'time':'t_dim'})  # TODO Do something with this or delete it
        
        return interpolated

    def construct_density( self, EOS='EOS10' ):
        
        '''
            Constructs the in-situ density using the salinity, temperture and 
            depth_0 fields and adds a density attribute to the t-grid dataset 
            
            Requirements: The supplied t-grid dataset must contain the 
            Practical Salinity and the Potential Temperature variables. The depth_0
            field must also be supplied. The GSW package is used to calculate
            The Absolute Pressure, Absolute Salinity and Conservate Temperature.
            
            Note that currently density can only be constructed using the EOS10
            equation of state.

        Parameters
        ----------
        EOS : equation of state, optional
            DESCRIPTION. The default is 'EOS10'.


        Returns
        -------
        None.
        adds attribute NEMO.dataset.density

        '''  
        debug(f"Constructing in-situ density for {get_slug(self)} with EOS \"{EOS}\"")
        try:
            if EOS != 'EOS10': 
                raise ValueError(str(self) + ': Density calculation for ' + EOS + ' not implemented.')
            if self.grid_ref != 't-grid':
                raise ValueError(str(self) + ': Density calculation can only be performed for a t-grid object,\
                                 the tracer grid for NEMO.' )
        
            try:    
                shape_ds = ( self.dataset.t_dim.size, self.dataset.z_dim.size, 
                                self.dataset.y_dim.size, self.dataset.x_dim.size )
                sal = self.dataset.salinity.to_masked_array()
                temp = self.dataset.temperature.to_masked_array()                
            except AttributeError:
                shape_ds = ( 1, self.dataset.z_dim.size, 
                                self.dataset.y_dim.size, self.dataset.x_dim.size )
                sal = self.dataset.salinity.to_masked_array()[np.newaxis,...]
                temp = self.dataset.temperature.to_masked_array()[np.newaxis,...]
            
            density = np.ma.zeros( shape_ds )
            
            s_levels = self.dataset.depth_0.to_masked_array()
            lat = self.dataset.latitude.values
            lon = self.dataset.longitude.values
            # Absolute Pressure 
            pressure_absolute = np.ma.masked_invalid(
                gsw.p_from_z( -s_levels, lat ) ) # depth must be negative    
            # Absolute Salinity            
            sal_absolute = np.ma.masked_invalid(
                gsw.SA_from_SP( sal, pressure_absolute, lon, lat ) )
            sal_absolute = np.ma.masked_less(sal_absolute,0)
            # Conservative Temperature
            temp_conservative = np.ma.masked_invalid(
                gsw.CT_from_pt( sal_absolute, temp ) )
            # In-situ density
            density = np.ma.masked_invalid( gsw.rho( 
                sal_absolute, temp_conservative, pressure_absolute ) )
            
            coords={'depth_0': (('z_dim','y_dim','x_dim'), self.dataset.depth_0.values),
                    'latitude': (('y_dim','x_dim'), self.dataset.latitude.values),
                    'longitude': (('y_dim','x_dim'), self.dataset.longitude.values)}
            dims=['z_dim', 'y_dim', 'x_dim']
            attributes = {'units': 'kg / m^3', 'standard name': 'In-situ density'}
            
            if shape_ds[0] != 1:
                coords['time'] = (('t_dim'), self.dataset.time.values)
                dims.insert(0, 't_dim')
    
            self.dataset['density'] = xr.DataArray( np.squeeze(density), 
                    coords=coords, dims=dims, attrs=attributes )
            
        except AttributeError as err:
            error(err)

    def trim_domain_size( self, dataset_domain ):
        """
        Trim the domain variables if the dataset object is a spatial subset
        
        Note: This breaks if the SW & NW corner values of nav_lat and nav_lon 
        are masked, as can happen if on land...
        """
        debug(f"Trimming {get_slug(self)} variables with {get_slug(dataset_domain)}")
        if (self.dataset['x_dim'].size != dataset_domain['x_dim'].size)  \
                or (self.dataset['y_dim'].size != dataset_domain['y_dim'].size):
            info(
                'The domain  and dataset objects are different sizes:'
                ' [{},{}] cf [{},{}]. Trim domain.'
                .format(
                    dataset_domain['x_dim'].size, dataset_domain['y_dim'].size,
                    self.dataset['x_dim'].size, self.dataset['y_dim'].size
                )
            )

            # Find the corners of the cut out domain.
            [j0,i0] = self.find_j_i_domain( self.dataset.nav_lat[0,0],
                                    self.dataset.nav_lon[0,0], dataset_domain )
            [j1,i1] = self.find_j_i_domain( self.dataset.nav_lat[-1,-1],
                                    self.dataset.nav_lon[-1,-1], dataset_domain )

            dataset_subdomain = dataset_domain.isel(
                                        y_dim = slice(j0, j1 + 1),
                                        x_dim = slice(i0, i1 + 1) )
            return dataset_subdomain
        else:
            return dataset_domain

    def copy_domain_vars_to_dataset( self, dataset_domain, grid_vars ):
        """
        Map the domain coordand metric variables to the dataset object.
        Expects the source and target DataArrays to be same sizes.
        """
        debug(f"Copying domain vars from {get_slug(dataset_domain)}/{get_slug(grid_vars)} to {get_slug(self)}")
        for var in grid_vars:
            try:
                new_name = self.var_mapping_domain[var]
                self.dataset[new_name] = dataset_domain[var].squeeze()
                debug("map: {} --> {}".format(var, new_name))
            except:  # FIXME Catch specific exception(s)
                pass  # TODO Should we log something here?

    def differentiate(self, in_varstr, dim='z_dim', out_varstr=None, out_obj=None):
        """
        Derivatives are computed in x_dim, y_dim, z_dim (or i,j,k) directions
        wrt lambda, phi, or z coordinates (with scale factor in metres not degrees).

        Derivatives are calculated using the approach adopted in NEMO,
        specifically using the 1st order accurate central difference
        approximation. For reference see section 3.1.2 (sec. Discrete operators)
        of the NEMO v4 Handbook.

        Currently the method does not accomodate all possible eventualities. It
        covers:
        1) d(grid_t)/dz --> grid_w
        
        Returns  an object (with the appropriate target grid_ref) containing
        derivative (out_varstr) as xr.DataArray
        
        This is hardwired to expect:
        1) depth_0 and e3_0 fields exist
        2) xr.DataArrays are 4D
        3) self.filename_domain if out_obj not specified
        4) If out_obj is not specified, one is built that is  the size of
            self.filename_domain. I.e. automatic subsetting of out_obj is not
            supported.
        
        Example usage:
        --------------
        # Initialise DataArrays
        nemo_t = coast.NEMO( fn_data, fn_domain, grid_ref='t-grid' )
        # Compute dT/dz
        nemo_w_1 = nemo_t.differentiate( 'temperature', dim='z_dim' )
        
        # For f(z)=-z. Compute df/dz = -1. Surface value is set to zero
        nemo_t.dataset['depth4D'],_ = xr.broadcast( nemo_t.dataset['depth_0'], nemo_t.dataset['temperature'] )
        nemo_w_4 = nemo_t.differentiate( 'depth4D', dim='z_dim', out_varstr='dzdz' )
        
        Provide an existing target NEMO object and target variable name:
        nemo_w_1 = nemo_t.differentiate( 'temperature', dim='z_dim', out_varstr='dTdz', out_obj=nemo_w_1 )
        
        
        Parameters
        ----------
        in_varstr : str, name of variable to differentiate
        dim : str, dimension to operate over. E.g. {'z_dim', 'y_dim', 'x_dim', 't_dim'}
        out_varstr : str, (optional) name of the target xr.DataArray
        out_obj : exiting NEMO obj to store xr.DataArray (optional)

        """
        #import xarray as xr

        new_units = ""

        # Check in_varstr exists in self.
        if hasattr( self.dataset, in_varstr ):
            # self.dataset[in_varstr] exists

            var = self.dataset[in_varstr] # for convenience

            nt = var.sizes['t_dim']
            nz = var.sizes['z_dim']
            ny = var.sizes['y_dim']
            nx = var.sizes['x_dim']

            ## Compute d(t_grid)/dz --> w-grid
            # Check grid_ref and dir. Determine target grid_ref.
            if (self.grid_ref == 't-grid') and (dim == 'z_dim'):
                out_grid = 'w-grid'

                # If out_obj exists check grid_ref, else create out_obj.
                if (out_obj is None) or (out_obj.grid_ref != out_grid):
                    try:
                        out_obj = NEMO( fn_domain=self.filename_domain, grid_ref=out_grid )
                    except:  # TODO Catch specific exception(s)
                        warn('Failed to create target NEMO obj. Perhaps self.',
                             'filename_domain={} is empty?'
                             .format(self.filename_domain))

                # Check is out_varstr is defined, else create it
                if out_varstr is None:
                    out_varstr = in_varstr + '_dz'

                # Create new DataArray with the same dimensions as the parent
                # Crucially have a coordinate value that is appropriate to the target location.
                blank = xr.zeros_like( var.isel(z_dim=[0]) ) # Using "z_dim=[0]" as a list preserves z-dimension
                blank.coords['depth_0'] -= blank.coords['depth_0'] # reset coord vals to zero
                # Add blank slice to the 'surface'. Concat over the 'dim' coords
                diff = xr.concat([blank, var.diff(dim)], dim)
                diff_ndim, e3w_ndim = xr.broadcast( diff, out_obj.dataset.e3_0.squeeze() )
                # Compute the derivative
                out_obj.dataset[out_varstr] = - diff_ndim / e3w_ndim

                # Assign attributes
                new_units = var.units+'/'+ out_obj.dataset.depth_0.units
                # Convert to a xr.DataArray and return
                out_obj.dataset[out_varstr].attrs = {
                                           'units': new_units,
                                           'standard_name': out_varstr}

                # Return in object.
                return out_obj

            else:
                warn('Not ready for that combination of grid ({}) and '
                     'derivative ({})'.format(self.grid_ref, dim))
                return None
        else:
            warn(f"{in_varstr} does not exist in {get_slug(self)} dataset")
            return None
        
    def apply_doodson_x0_filter(self, var_str):
        ''' Applies Doodson X0 filter to a variable. 
    
        Input variable is expected to be hourly.
        Output is saved back to original dataset as {var_str}_dxo
        
        !!WARNING: Will load in entire variable to memory. If dataset large,
        then subset before using this method or ensure you have enough free 
        RAM to hold the variable (twice). 
        
        DB:: Currently not tested in unit_test.py'''
        var = self.dataset[var_str]
        new_var_str = var_str + '_dx0'
        old_dims = var.dims
        time_index = old_dims.index('t_dim')
        filtered = stats_util.doodson_x0_filter(var, ax=time_index)
        if filtered is not None:
            self.dataset[new_var_str] = (old_dims, filtered)
        return
    
    def harmonics_combine(self, constituents, components = ['x','y']):
        '''
        Contains a new NEMO object containing combined harmonic information
        from the original object. 
        
        NEMO saves harmonics to individual variables such as M2x, M2y... etc. 
        This routine will combine these variables (depending on constituents)
        into a single data array. This new array will have the new dimension
        'constituent' and a new data coordinate 'constituent_name'.
        
        Parameters
        ----------
        constituents : List of strings containing constituent names to combine.
                       The case of these strings should match that used in 
                       NEMO output. If a constituent is not found, no problem,
                       it just won't be in the combined dataset.
        components   : List of strings containing harmonic components to look
                       for. By default, this looks for the complex components
                       'x' and 'y'. E.g. if constituents = ['M2'] and
                       components is left as default, then the routine looks
                       for ['M2x', and 'M2y']. 

        Returns
        -------
        NEMO() object, containing combined harmonic variables in a new dataset.
        '''
        
        # Select only the specified constituents. NEMO model harmonics names are
        # things like "M2x" and "M2y". Ignore current harmonics. Start by constructing
        # the possible variable names
        names_x = np.array([cc + components[0] for cc in constituents])
        names_y = np.array([cc + components[1] for cc in constituents])
        constituents = np.array(constituents, dtype='str')
        
        # Compare against names in file
        var_keys = np.array(list(self.dataset.keys()))
        indices = [np.where( names_x == ss) for ss in names_x if ss in var_keys]
        indices = np.array(indices).T.squeeze()
        
        # Index the possible names to match file names
        print(indices)
        names_x = names_x[indices]
        names_y = names_y[indices]
        constituents = constituents[indices]
        
        # Concatenate x and y variables into one array
        x_arrays = [self.dataset[ss] for ss in names_x]
        harmonic_x = 'harmonic_'+components[0]
        x_data = xr.concat(x_arrays, dim = 'constituent').rename(harmonic_x)
        y_arrays = [self.dataset[ss] for ss in names_y]
        harmonic_y = 'harmonic_'+components[1]
        y_data = xr.concat(y_arrays, dim = 'constituent').rename(harmonic_y)
        
        nemo_harmonics = NEMO()
        nemo_harmonics.dataset = xr.merge([x_data, y_data])
        nemo_harmonics.dataset['constituent'] = constituents
        
        return nemo_harmonics
        
    def harmonics_convert(self, direction='cart2polar',
                          x_var='harmonic_x', y_var='harmonic_y',
                          a_var='harmonic_a', g_var='harmonic_g',
                          degrees=True):
        '''
        Converts NEMO harmonics from cartesian to polar or vice versa.
        Make sure this NEMO object contains combined harmonic variables
        obtained using harmonics_combine().
        
        *Note:
        
        Parameters
        ----------
        direction (str) : Choose 'cart2polar' or 'polar2cart'. If 'cart2polar'
                          Then will look for variables x_var and y_var. If 
                          polar2cart, will look for a_var (amplitude) and 
                          g_var (phase).
        x_var (str)     : Harmonic x variable name in dataset (or output)
                          default = 'harmonic_x'.
        y_var (str)     : Harmonic y variable name in dataset (or output)
                          default = 'harmonic_y'.
        a_var (str)     : Harmonic amplitude variable name in dataset (or output)
                          default = 'harmonic_a'.
        g_var (str)     : Harmonic phase variable name in dataset (or output)
                          default = 'harmonic_g'.
        degrees (bool)  : Whether input/output phase are/will be in degrees.
                          Default is True.

        Returns
        -------
        Modifies NEMO() dataset in place. New variables added.
        '''
        if direction == 'cart2polar':
            a,g = general_utils.cart2polar(self.dataset[x_var], 
                                           self.dataset[y_var], 
                                           degrees=degrees)
            self.dataset[a_var] = a
            self.dataset[g_var] = g
        elif direction == 'polar2cart':
            x,y = general_utils.polar2cart(self.dataset[a_var], 
                                           self.dataset[g_var],
                                           degrees=degrees)
            self.dataset[x_var] = x
            self.dataset[y_var] = y
        else:
            print('Unknown direction setting. Choose cart2polar or polar2cart')
        
        return

        
        
        
        
    
                
        
        
