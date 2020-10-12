from .COAsT import COAsT
from .NEMO import NEMO
from scipy.ndimage import convolve1d
from scipy import interpolate
import gsw
import xarray as xr
import numpy as np
from scipy.integrate import cumtrapz
from .logging_util import get_slug, debug, error
import warnings
import traceback

# =============================================================================
# The TRANSECT module is a place for code related to transects only
# =============================================================================

class Transect:
    GRAVITY = 9.8 # m s^-2
    EARTH_ROT_RATE = 7.2921 * 10**(-5) # rad/s
    
    
    @staticmethod
    def moving_average(array_to_smooth, window=2, axis=-1):
        '''
        Returns the input array smoothed along the given axis using convolusion
        '''
        debug(f"Fetching moving average for {array_to_smooth}")
        return convolve1d( array_to_smooth, np.ones(window), axis=axis ) / window


    @staticmethod
    def interpolate_slice(variable_slice, depth, interpolated_depth=None ):
        '''
        Linearly interpolates the variable at a single time along the z_dim, which must be the
        first axis.

        Parameters
        ----------
        variable_slice : Variable to interpolate (z_dim, r_dim)
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
            
        return interpolated_depth_variable_slice, interpolated_depth 
    
    
    @staticmethod
    def gen_z_levels(max_depth):
        ''' Generates a pre-defined 1d vertical depth coordinate,
        i.e. horizontal z-level vertical coordinates up to a supplied 
        maximum depth, 'max_depth' 
        
        Parameters
        ----------
        max_depth : int, bottom level depth 
        '''
        
        max_depth = max_depth + 650
        z_levels_0_50 = np.arange(0,55,5)
        z_levels_60_290 = np.arange(60,300,10)
        z_levels_300_600 = np.arange(300,650,50)
        z_levels_650_ = np.arange(650,max_depth+150,150)
        z_levels = np.concatenate( (z_levels_0_50, z_levels_60_290, 
                                    z_levels_300_600, z_levels_650_) )
        z_levels = z_levels[z_levels <= max_depth] 
        return z_levels
    
    
    def __init__(self, nemo: COAsT, point_A: tuple=None, point_B: tuple=None, y_indices=None, x_indices=None):
        '''
        Class defining a generic transect type, which is a 3d dataset along 
        a linear path between a point A and a point B, with a time dimension,
        a depth dimension and an along transect dimension. 
        The model Data on the supplied grid is subsetted in its entirety along these dimensions.
        
        Note that Point A should be closer to the southern boundary of the model domain.
        
        The user can either supply the start and end (lat,lon) coordinates of the
        transect, point_A and point_B respectively, or the model y, x indices defining it.
        In the latter case the user must ensure that the indices define a continuous
        transect, e.g. y=[10,11,11,12], x=[5,5,6,6].
        Only limited checks are performed on the suitability of the indices.
        
        Example usage:
            point_A = (54,-15)
            point_B = (56,-12)
            transect = coast.Transect( nemo_t, point_A, point_B )
            or 
            transect = coast.Transect( nemo_f, y_indices=y_ind, x_indices=x_ind )
            
        Parameters
        ----------
        nemo : NEMO object 
        point_A : tuple, (lat,lon)
        point_B : tuple, (lat,lon)
        y_indices : 1d array of model y indices defining the points of the transect 
        x_indices : 1d array of model x indices defining the points of the transect 

        '''
        debug(f"Creating a new {get_slug(self)}")
        try:
            self.filename_domain = nemo.filename_domain
            
            if point_A is not None and point_B is not None:
                # point A should be of lower latitude than point B
                if abs(point_B[0]) < abs(point_A[0]):
                    self.point_A = point_B
                    self.point_B = point_A
                else:
                    self.point_A = point_A
                    self.point_B = point_B
                   
                # Get points on transect    
                tran_y_ind, tran_x_ind, tran_len = nemo.transect_indices(self.point_A, self.point_B)
                tran_y_ind, tran_x_ind = self.process_transect_indices( nemo, \
                                        np.asarray(tran_y_ind), np.asarray(tran_x_ind) )
            elif y_indices is not None and x_indices is not None:
                if y_indices[0] > y_indices[-1]:
                    y_indices = y_indices[::-1]
                    x_indices = x_indices[::-1]
                tran_y_ind, tran_x_ind = self.process_transect_indices( nemo, \
                                        y_indices, x_indices )
                self.point_A = (nemo.dataset.latitude[tran_y_ind[0],tran_x_ind[0]],
                                nemo.dataset.longitude[tran_y_ind[0],tran_x_ind[0]])
                self.point_B = (nemo.dataset.latitude[tran_y_ind[-1],tran_x_ind[-1]],
                                nemo.dataset.longitude[tran_y_ind[-1],tran_x_ind[-1]])
            else:
                raise ValueError("Must supply both point_A and point_B of transect \
                                 or the indices defining it.")
                
            # indices along the transect        
            self.y_ind = tran_y_ind
            self.x_ind = tran_x_ind
            self.len = len(tran_y_ind)
            self.data_cross_tran_flow = xr.Dataset()
            
            # Subset the nemo data along the transect creating a new dimension (r_dim),
            # which is a paramterisation for x_dim and y_dim defining the transect
            da_tran_y_ind = xr.DataArray( tran_y_ind, dims=['r_dim'])
            da_tran_x_ind = xr.DataArray( tran_x_ind, dims=['r_dim'])
            self.data = nemo.dataset.isel(y_dim = da_tran_y_ind, x_dim = da_tran_x_ind)
    
            debug(f"{get_slug(self)} initialised")
        except ValueError:
            print(traceback.format_exc())


    def process_transect_indices(self, nemo, tran_y_ind, tran_x_ind):
        '''
        Get the transect indices on a specific grid

        Parameters
        ----------
        nemo : the model grid to define the transect on
        
        Return
        ----------
        tran_y_ind : array of y_dim indices
        tran_x_ind : array of x_dim indices

        '''
        debug(f"Fetching transect indices for {get_slug(self)} with {get_slug(nemo)}")
        try:
            # Redefine transect so that each point on the transect is seperated
            # from its neighbours by a single index change in y or x, but not both
            dist_option_1 = nemo.dataset.e2.values[tran_y_ind, tran_x_ind] + nemo.dataset.e1.values[tran_y_ind+1, tran_x_ind]
            dist_option_2 = nemo.dataset.e2.values[tran_y_ind, tran_x_ind+1] + nemo.dataset.e1.values[tran_y_ind, tran_x_ind]
            spacing = np.abs( np.diff(tran_y_ind) ) + np.abs( np.diff(tran_x_ind) )
            if spacing.max() > 2:
                raise ValueError("The transect is not continuous. The transect must be defined on " 
                                     "adjacent grid points.")
            spacing[spacing!=2]=0
            doublespacing = np.nonzero( spacing )[0]
            for ispacing in doublespacing[::-1]:
                if dist_option_1[ispacing] < dist_option_2[ispacing]:
                    tran_y_ind = np.insert( tran_y_ind, ispacing+1, tran_y_ind[ispacing+1] )
                    tran_x_ind = np.insert( tran_x_ind, ispacing+1, tran_x_ind[ispacing] )
                else:
                    tran_y_ind = np.insert( tran_y_ind, ispacing+1, tran_y_ind[ispacing] )
                    tran_x_ind = np.insert( tran_x_ind, ispacing+1, tran_x_ind[ispacing+1] ) 
            return tran_y_ind, tran_x_ind
        except ValueError:
            print(traceback.format_exc())
        
     
    def plot_transect_on_map(self):
        '''
        Plot transect location on a map
        
        Example usage:
        --------------
        tran = coast.Transect( (54,-15), (56,-12), nemo )
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

        cset = plt.plot(self.data.longitude, self.data.latitude, c='k')

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

    

class Transect_f(Transect):
    '''
    Class defining a transect on the f-grid, which is a 3d dataset along 
    a linear path between a point A and a point B, with a time dimension,
    a depth dimension and an along transect dimension. The model Data on f-grid 
    is subsetted in its entirety along these dimensions.
    
    Note that Point A should be closer to the southern boundary of the model domain.
    
    The user can either supply the start and end (lat,lon) coordinates of the
    transect, point_A and point_B respectively, or the model y, x indices defining it.
    In the latter case the user must ensure that the indices define a continuous
    transect, e.g. y=[10,11,11,12], x=[5,5,6,6].
    Only limited checks are performed on the suitability of the indices.
    
    Example usage:
        point_A = (54,-15)
        point_B = (56,-12)
        transect = coast.Transect_f( nemo_f, point_A, point_B )
        or 
        transect = coast.Transect_f( nemo_f, y_indices=y_ind, x_indices=x_ind )
        
    Parameters
    ----------
    nemo_f : NEMO object on the f-grid
    point_A : tuple, (lat,lon)
    point_B : tuple, (lat,lon)
    y_indices : 1d array of model y indices defining the points of the transect 
    x_indices : 1d array of model x indices defining the points of the transect 

    '''
    
    def __init__(self, nemo_f: COAsT, point_A: tuple=None, point_B: tuple=None, 
                 y_indices=None, x_indices=None):
        super().__init__(nemo_f, point_A, point_B, y_indices, x_indices)
        
        
    def calc_flow_across_transect(self, nemo_u: COAsT, nemo_v: COAsT):
        """
    
        Computes the flow through the transect at each segment and creates a new 
        dataset 'Transect_f.data_cross_tran_flow' defined on the normal velocity
        points along the transect.
        Transect normal velocities are calculated at each grid point and stored in
        in Transect_f.data_cross_tran_flow.normal_velocities,
        Depth integrated volume transport across the transect is calculated 
        at each transect segment and stored in Transect_f.data_cross_tran_flow.normal_transports
        The latitude, longitude and the horizontal and vertical scale factors
        on the normal velocity points are also stored in the dataset.
        
        parameters
        ----------
        nemo_u : Nemo object on the u-grid containing the i-component velocities
        nemo_v : Nemo object on the v-gridc ontaining the j-component velocities
        
        """
        debug(f"Computing flow across the transect for {get_slug(self)}")
                
        # subset the u and v datasets 
        da_y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] )
        da_x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] )
        u_ds = nemo_u.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        v_ds = nemo_v.dataset.isel(y_dim = da_y_ind, x_dim = da_x_ind)
        
        # If there is no time dimension, add one. This is so
        # indexing can assume a time dimension exists
        if 't_dim' not in u_ds.dims:
            u_ds = u_ds.expand_dims(dim={'t_dim':1},axis=0)
        if 't_dim' not in v_ds.dims:
            v_ds = v_ds.expand_dims(dim={'t_dim':1},axis=0)
        
        velocity = np.ma.zeros( (u_ds.t_dim.size, u_ds.z_dim.size, u_ds.r_dim.size-1) )
        vol_transport = np.ma.zeros( (u_ds.t_dim.size, u_ds.z_dim.size, u_ds.r_dim.size-1) )
        depth_0 = np.ma.zeros( (u_ds.z_dim.size, u_ds.r_dim.size-1) )
        latitude = np.ma.zeros( (u_ds.r_dim.size-1) )
        longitude = np.ma.zeros( (u_ds.r_dim.size-1) )
        e1 = np.ma.zeros( (u_ds.r_dim.size-1) )
        e2 = np.ma.zeros( (u_ds.r_dim.size-1) )
        e3_0 = np.ma.zeros( (u_ds.z_dim.size, u_ds.r_dim.size-1) )
        
        # Find the indices where the derivative of the transect in the north, south, east and west
        # directions are positive.
        dr_n = np.where(np.diff(self.y_ind)>0, np.arange(0,self.data.r_dim.size-1), np.nan )
        dr_n = dr_n[~np.isnan(dr_n)].astype(int)
        dr_e = np.where(np.diff(self.x_ind)>0, np.arange(0,self.data.r_dim.size-1), np.nan )
        dr_e = dr_e[~np.isnan(dr_e)].astype(int)
        dr_w = np.where(np.diff(self.x_ind)<0, np.arange(0,self.data.r_dim.size-1), np.nan )   
        dr_w = dr_w[~np.isnan(dr_w)].astype(int)
        
        # u flux (+ in)
        velocity[:,:,dr_n] = u_ds.vozocrtx.to_masked_array()[:,:,dr_n+1]
        vol_transport[:,:,dr_n] = ( velocity[:,:,dr_n] * u_ds.e2.to_masked_array()[dr_n+1] *
                                          u_ds.e3_0.to_masked_array()[:,dr_n+1] )
        depth_0[:,dr_n] = u_ds.depth_0.to_masked_array()[:,dr_n+1]
        latitude[dr_n] = u_ds.latitude.values[dr_n+1]
        longitude[dr_n] = u_ds.longitude.values[dr_n+1]
        e1[dr_n] = u_ds.e1.values[dr_n+1]
        e2[dr_n] = u_ds.e2.values[dr_n+1]
        e3_0[:,dr_n] = u_ds.e3_0.values[:,dr_n+1]
        
        # v flux (- in) 
        velocity[:,:,dr_e] = - v_ds.vomecrty.to_masked_array()[:,:,dr_e+1]
        vol_transport[:,:,dr_e] = ( velocity[:,:,dr_e] * v_ds.e1.to_masked_array()[dr_e+1] *
                                  v_ds.e3_0.to_masked_array()[:,dr_e+1] )
        depth_0[:,dr_e] = v_ds.depth_0.to_masked_array()[:,dr_e+1]
        latitude[dr_e] = v_ds.latitude.values[dr_e+1]
        longitude[dr_e] = v_ds.longitude.values[dr_e+1]
        e1[dr_e] = v_ds.e1.values[dr_e+1]
        e2[dr_e] = v_ds.e2.values[dr_e+1]
        e3_0[:,dr_e] = v_ds.e3_0.values[:,dr_e+1]
        
        # v flux (+ in)
        velocity[:,:,dr_w] = v_ds.vomecrty.to_masked_array()[:,:,dr_w]
        vol_transport[:,:,dr_w] = ( velocity[:,:,dr_w] * v_ds.e1.to_masked_array()[dr_w] *
                                  v_ds.e3_0.to_masked_array()[:,dr_w] )
        depth_0[:,dr_w] = v_ds.depth_0.to_masked_array()[:,dr_w]
        latitude[dr_w] = v_ds.latitude.values[dr_w]
        longitude[dr_w] = v_ds.longitude.values[dr_w]
        e1[dr_w] = v_ds.e1.values[dr_w]
        e2[dr_w] = v_ds.e2.values[dr_w]
        e3_0[:,dr_w] = v_ds.e3_0.values[:,dr_w]
           
        # Add DataArrays to dataset           
        self.data_cross_tran_flow['normal_velocities'] = xr.DataArray( np.squeeze(velocity), 
                    coords={'time': (('t_dim'), u_ds.time.values),'depth_0': (('z_dim','r_dim'), depth_0)
                            ,'latitude': (('r_dim'), latitude), 'longitude': (('r_dim'), longitude) },
                    dims=['t_dim', 'z_dim', 'r_dim'] )            
        self.data_cross_tran_flow['normal_transports'] \
                    = xr.DataArray( np.squeeze(np.sum(vol_transport, axis=1)) / 1000000.,
                    coords={'time': (('t_dim'), u_ds.time.values)
                            ,'latitude': (('r_dim'), latitude), 'longitude': (('r_dim'), longitude)},
                    dims=['t_dim', 'r_dim'] )          
        self.data_cross_tran_flow['e1'] = xr.DataArray( e1, dims=['r_dim'] ) 
        self.data_cross_tran_flow['e2'] = xr.DataArray( e2, dims=['r_dim'] ) 
        self.data_cross_tran_flow['e3_0'] = xr.DataArray( e3_0, dims=['z_dim','r_dim'] ) 
        # DataArray attributes   
        self.data_cross_tran_flow.normal_velocities.attrs['units'] = 'm/s'
        self.data_cross_tran_flow.normal_velocities.attrs['standard_name'] = 'velocity across the transect'
        self.data_cross_tran_flow.normal_velocities.attrs['long_name'] = 'velocity across the transect defined on the normal velocity grid points'    
        self.data_cross_tran_flow.normal_transports.attrs['units'] = 'Sv'
        self.data_cross_tran_flow.normal_transports.attrs['standard_name'] = 'depth integrated volume transport across transect'
        self.data_cross_tran_flow.normal_transports.attrs['long_name'] = 'depth integrated volume transport across the transect defined on the normal velocity grid points' 
        self.data_cross_tran_flow.depth_0.attrs['units'] = 'm'
        self.data_cross_tran_flow.depth_0.attrs['standard_name'] = 'depth'
        self.data_cross_tran_flow.depth_0.attrs['long_name'] = 'Initial depth at time zero defined at the normal velocity grid points'
                        
  
    def __pressure_grad_fpoint(self, ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, r_ind, velocity_component):
        """
        Calculates the hydrostatic and surface pressure gradients at a set of f-points
        along the transect, i.e. at a set of specific values of r_dim (but for all time and depth).
        The caller must supply four datasets that contain the variables which define
        the hydrostatic and surface pressure at all vertical z_levels and all time 
        on the t-points around the transect i.e. for a set of f-points on the transect 
        defined at (j+1/2, i+1/2), t-points are supplied at (j,i), (j+1,i), (j,i+1), (j+1,i+1), 
        corresponding to ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, respectively. 
        
        The velocity_component defines whether u or v is normal to the transect 
        for the segments of the transect. A segment of transect is 
        defined as being r_dim to r_dim+1 where r_dim is the along transect dimension.

        Parameters
        ----------
        ds_T : coast.Transect_t on y=self.y_ind, x=self.x_ind
        ds_T_j1 : coast.Transect_t on y=self.y_ind+1, x=self.x_ind
        ds_T_i1 : coast.Transect_t on y=self.y_ind, x=self.x_ind+1
        ds_T_j1i1 : coast.Transect_t on y=self.y_ind+1, x=self.x_ind+1
        r_ind: 1d array, along transect indices 
        velocity_component : str, normal velocity at r_ind
        
        Returns
        -------
        hpg_f : DataArray with dimensions in time and depth and along transect
            hydrostatic pressure gradient at a set of f-points along the transect
            for all time and depth
        spg_f : DataArray with dimensions in time and depth and along transect
            surface pressure gradient at a set of f-points along the transect

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
    
  
    def calc_geostrophic_flow(self, nemo_t: COAsT, ref_density=None):
        """
        This method will calculate the geostrophic velocity and volume transport
        (due to the geostrophic current) across the transect. 
        4 variables are added to the Transect_f.data_cross_tran_flow dataset:
            1. normal_velocity_hpg      (t_dim, depth_z_levels, r_dim)
            This is the velocity due to the hydrostatic pressure gradient
            2. normal_velocity_spg      (t_dim, r_dim)
            This is the velocity due to the surface pressure gradient
            3. normal_transport_hpg  (t_dim, r_dim)
            This is the volume transport due to the hydrostatic pressure gradient
            4. normal_transport_AB_spg  (t_dim, r_dim
            This is the volume transport due to the surface pressure gradient
                                                                       
        The implementation works by regridding from the native vertical grid to 
        horizontal z_levels in order to perform the horizontal gradients.
        Currently the level depths are assumed fixed at their initial depths,
        i.e. at time zero.
        
        Parameters
        ----------
        nemo_t : COAsT
            This is the nemo model data on the t-grid for the entire domain. It
            must contain the temperature, salinity and t-grid domain data (e1t, e2t, e3t_0).
        ref_density : float, optional
            reference density value. If None a transect mean density will be calculated 
            and used.


        """
        debug(f"Calculating geostrophic velocity and volume transport for {get_slug(self)} with "
              f"{get_slug(nemo_t)}")

        # If there is no time dimension, add one then remove at end. This is so
        # indexing can assume a time dimension exists
        nemo_t_local = nemo_t.copy()
        if 't_dim' not in nemo_t_local.dataset.dims:
            nemo_t_local.dataset = nemo_t_local.dataset.expand_dims(dim={'t_dim':1},axis=0)
            
        #We need to calculate the pressure at four t-points to get an
        # average onto the pressure gradient at the f-points, which will then
        # be averaged onto the normal velocity points. Here we subset the nemo_t 
        # data around the transect so we have these four t-grid points at each 
        # point along the transect        
        tran_t = Transect_t(nemo_t_local, y_indices=self.y_ind, x_indices=self.x_ind)            # j,i
        tran_t_j1 = Transect_t(nemo_t_local, y_indices=self.y_ind+1, x_indices=self.x_ind)       # j+1,i
        tran_t_i1 = Transect_t(nemo_t_local, y_indices=self.y_ind, x_indices=self.x_ind+1)       # j,i+1
        tran_t_j1i1 = Transect_t(nemo_t_local, y_indices=self.y_ind+1, x_indices=self.x_ind+1)   # j+1,i+1
        
        bath_max = np.max([tran_t.data.bathymetry.max().item(), 
                           tran_t_j1.data.bathymetry.max().item(),
                           tran_t_i1.data.bathymetry.max().item(), 
                           tran_t_j1i1.data.bathymetry.max().item()])
                
        z_levels = Transect.gen_z_levels(bath_max) 
        
        tran_t.construct_pressure(ref_density, z_levels, extrapolate=True)
        tran_t_j1.construct_pressure(ref_density, z_levels, extrapolate=True)
        tran_t_i1.construct_pressure(ref_density, z_levels, extrapolate=True)
        tran_t_j1i1.construct_pressure(ref_density, z_levels, extrapolate=True)   
                  
        # Remove the mean hydrostatic pressure on each z_level from the hydrostatic pressure.
        # This helps to reduce the noise when taking the horizontal gradients of hydrostatic pressure.
        # Also catch and ignore nan-slice warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pressure_h_zlevel_mean = xr.concat( (tran_t.data.pressure_h_zlevels, tran_t_j1.data.pressure_h_zlevels, 
                                 tran_t_i1.data.pressure_h_zlevels, tran_t_j1i1.data.pressure_h_zlevels), 
                                 dim='concat_dim' ).mean(dim=('concat_dim','r_dim','t_dim'),skipna=True)
            
            if ref_density is None:
                ref_density = ( xr.concat( (tran_t.data.density_zlevels, 
                            tran_t_j1.data.density_zlevels, 
                            tran_t_i1.data.density_zlevels, 
                            tran_t_j1i1.data.density_zlevels), dim='concat_dim' )
                            .mean(dim=('concat_dim','r_dim','t_dim','depth_z_levels'),skipna=True).item() )
                
        tran_t.data['pressure_h_zlevels'] = tran_t.data.pressure_h_zlevels - pressure_h_zlevel_mean
        tran_t_j1.data['pressure_h_zlevels'] = tran_t_j1.data.pressure_h_zlevels - pressure_h_zlevel_mean
        tran_t_i1.data['pressure_h_zlevels'] = tran_t_i1.data.pressure_h_zlevels - pressure_h_zlevel_mean
        tran_t_j1i1.data['pressure_h_zlevels'] = tran_t_j1i1.data.pressure_h_zlevels - pressure_h_zlevel_mean
                
        # Coriolis parameter
        f = 2 * self.EARTH_ROT_RATE * np.sin( np.deg2rad(self.data.latitude) )
        
        # Find the indices where the derivative of the transect in the north, south, east and west
        # directions are positive.
        dr_n = np.where(np.diff(self.y_ind)>0, np.arange(0,self.data.r_dim.size-1), np.nan )
        dr_e = np.where(np.diff(self.x_ind)>0, np.arange(0,self.data.r_dim.size-1), np.nan )
        dr_w = np.where(np.diff(self.x_ind)<0, np.arange(0,self.data.r_dim.size-1), np.nan )        
        dr_list = [dr_n[~np.isnan(dr_n)].astype(int),
                   dr_e[~np.isnan(dr_e)].astype(int), dr_w[~np.isnan(dr_w)].astype(int)]
        
        # horizontal scale factors on the relevent u and v grids that are
        # normal to the transect for dr_n, dr_e, dr_w
        e2u_j1  = 0.5 * ( tran_t_j1.data.e2.data[dr_list[0]] + tran_t_j1i1.data.e2.data[dr_list[0]] )
        e1v_i1  = 0.5 * ( tran_t_i1.data.e1.data[dr_list[1]] + tran_t_j1i1.data.e1.data[dr_list[1]] )
        e1v     = 0.5 * ( tran_t.data.e1.data[dr_list[2]] + tran_t_j1.data.e1.data[dr_list[2]] )
        e_horiz_vel = [e2u_j1, e1v_i1, e1v] 
        # Horizontal scale factors on f-grid for dr_n, dr_e, dr_w
        e_horiz_f   = [self.data.e2,  
                       self.data.e1, self.data.e1]
        # velocity component normal to transect for dr_n, dr_s, dr_e, dr_w
        velocity_component = ["u","v","v"]
        # Geostrophic flow direction across transect
        flow_direction = [-1,-1,1]   
        
        # The cross transect flow is defined on the u and v points that are across
        # the transect, i.e. between f points, therefore the attributes of the
        # data_cross_flow dataset need to be on these points.
        da_y_ind = xr.DataArray( self.y_ind, dims=['r_dim'] )
        da_x_ind = xr.DataArray( self.x_ind, dims=['r_dim'] )
        u_ds = NEMO( fn_domain=self.filename_domain, grid_ref='u-grid' ).dataset \
            .isel(y_dim=da_y_ind, x_dim=da_x_ind)
        v_ds = NEMO( fn_domain=self.filename_domain, grid_ref='v-grid' ).dataset \
            .isel(y_dim=da_y_ind, x_dim=da_x_ind)
        ds = [u_ds,v_ds,v_ds]
        
        # Drop the last point because the normal velocity points are defined at
        # the middle of a segment and there is as a result one less point.
        normal_velocity_hpg = np.zeros_like(tran_t.data.pressure_h_zlevels)[:,:,:-1] 
        normal_velocity_spg = np.zeros_like(tran_t.data.pressure_s)[:,:-1]
        latitude = np.zeros( (u_ds.r_dim.size-1) )
        longitude = np.zeros( (u_ds.r_dim.size-1) )
        depth_0 = np.ma.zeros( (u_ds.z_dim.size, u_ds.r_dim.size-1) )
        # horizontal scale factors for each segmant of transect
        e_horiz = np.zeros( (tran_t.data.t_dim.size, tran_t.data.r_dim.size-1) ) 
        # Contruct geostrophic flow
        for dr, vel_comp, flow_dir, e_hor_vel, e_hor_f, i_ds in \
                zip(dr_list, velocity_component, flow_direction, e_horiz_vel, e_horiz_f, ds) :
            hpg, spg        = self.__pressure_grad_fpoint( tran_t.data, 
                                tran_t_j1.data, tran_t_i1.data, 
                                tran_t_j1i1.data, dr, vel_comp )
            hpg_r1, spg_r1  = self.__pressure_grad_fpoint( tran_t.data,
                                tran_t_j1.data,
                                tran_t_i1.data, 
                                tran_t_j1i1.data, dr+1, vel_comp )
            normal_velocity_hpg[:,:,dr] = ( flow_dir * 0.5 * (e_hor_f.data[dr]*hpg/f.data[dr] 
                                            + e_hor_f.data[dr+1]*hpg_r1/f.data[dr+1]) 
                                            / (e_hor_vel * ref_density) )
            normal_velocity_spg[:,dr]   = ( flow_dir * 0.5 * (e_hor_f.data[dr]*spg/f.data[dr] 
                                            + e_hor_f.data[dr+1]*spg_r1/f.data[dr+1]) 
                                            / (e_hor_vel * ref_density) ) 
            e_horiz[:,dr] = e_hor_vel
            depth_0[:,dr] = i_ds.depth_0.to_masked_array()[:,dr]
            latitude[dr]  = i_ds.latitude.data[dr]
            longitude[dr] = i_ds.longitude.data[dr]
            
        # Bathymetry at normal velocity points             
        #H = np.zeros_like( self.data.bathymetry.values )[:-1]
        H = 0.5*(self.data.bathymetry.values[:-1] + self.data.bathymetry.values[1:])
        # Remove redundent levels below bathymetry
        normal_velocity_hpg = np.where( z_levels[:,np.newaxis] <= H, 
                                       normal_velocity_hpg, np.nan )           
        active_z_levels = np.count_nonzero(~np.isnan(normal_velocity_hpg),axis=1).max() 
        normal_velocity_hpg = normal_velocity_hpg[:,:active_z_levels,:]
        z_levels = z_levels[:active_z_levels]
        
        # DataArray attributes
        coords_hpg={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), latitude),
                'longitude': (('r_dim'), longitude)}
        dims_hpg=['depth_z_levels', 'r_dim']
        attributes_hpg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the hydrostatic pressure gradient'}
        coords_spg={'latitude': (('r_dim'), latitude),
                'longitude': (('r_dim'), longitude)}
        dims_spg=['r_dim']
        attributes_spg = {'units': 'm/s', 'standard name': 'velocity across the \
                          transect due to the surface pressure gradient'}
                
        # Add time if required
        if 't_dim' in tran_t.data.dims:
            coords_hpg['time'] = (('t_dim'), tran_t.data.time)
            dims_hpg.insert(0, 't_dim')
            coords_spg['time'] = (('t_dim'), tran_t.data.time)
            dims_spg.insert(0, 't_dim')
        
        # Add DataArrays  to dataset
        self.data_cross_tran_flow['normal_velocity_hpg'] = xr.DataArray( np.squeeze(normal_velocity_hpg),
                coords=coords_hpg, dims=dims_hpg, attrs=attributes_hpg)
        self.data_cross_tran_flow['normal_velocity_spg'] = xr.DataArray( np.squeeze(normal_velocity_spg),
                coords=coords_spg, dims=dims_spg, attrs=attributes_spg)
        self.data_cross_tran_flow['normal_transport_hpg'] = ( self.data_cross_tran_flow
                .normal_velocity_hpg.fillna(0).integrate(dim='depth_z_levels') ) * e_horiz / 1000000   
        self.data_cross_tran_flow.normal_transport_hpg.attrs = {'units': 'Sv', 
                'standard_name': 'volume transport across transect due to the hydrostatic pressure gradient'}        
        self.data_cross_tran_flow['normal_transport_spg'] = self.data_cross_tran_flow.normal_velocity_spg * H * e_horiz / 1000000
        self.data_cross_tran_flow.normal_transport_spg.attrs = {'units': 'Sv', 
                'standard_name': 'volume transport across transect due to the surface pressure gradient'}       
                    
        self.data_cross_tran_flow['latitude'] = xr.DataArray( latitude, dims=['r_dim'] ) 
        self.data_cross_tran_flow['longitude'] = xr.DataArray( longitude, dims=['r_dim'] ) 
        self.data_cross_tran_flow['e12'] = xr.DataArray( e_horiz[0,:], dims=['r_dim'] ) 
        self.data_cross_tran_flow['depth_0_original'] = xr.DataArray( depth_0, dims=['z_dim','r_dim'] ) 
        self.data_cross_tran_flow.depth_0_original.attrs['units'] = 'm'
        self.data_cross_tran_flow.depth_0_original.attrs['standard_name'] = 'original depth coordinate'
        self.data_cross_tran_flow.e12.attrs['standard_name'] = \
                'horizontal scale factor along the transect at the normal velocity point'


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
            data = self.data_cross_tran_flow.sel(t_dim = time)
        except KeyError:
            data = self.data_cross_tran_flow.isel(t_dim = time)        
        
        if smoothing_window != 0:
            normal_velocities, depth = Transect.interpolate_slice( data.normal_velocities, data.depth_0 )            
            normal_velocities = Transect.moving_average(normal_velocities, smoothing_window, axis=-1)
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
            data = self.data_cross_tran_flow.sel(t_dim = time)
        except KeyError:
            data = self.data_cross_tran_flow.isel(t_dim = time)            
        
        if smoothing_window != 0:    
            transport = self.moving_average(data.normal_transports, smoothing_window, axis=-1)
        else:
            transport = data.normal_transports
        
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
 
    
class Transect_t(Transect):
    '''
    Class defining a transect on the t-grid, which is a 3d dataset along 
    a linear path between a point A and a point B, with a time dimension,
    a depth dimension and an along transect dimension. The model Data on t-grid 
    is subsetted in its entirety along these dimensions.
    
    Note that Point A should be closer to the southern boundary of the model domain.
    
    The user can either supply the start and end (lat,lon) coordinates of the
    transect, point_A and point_B respectively, or the model y, x indices defining it.
    In the latter case the user must ensure that the indices define a continuous
    transect, e.g. y=[10,11,11,12], x=[5,5,6,6].
    Only limited checks are performed on the suitability of the indices.
    
    Example usage:
        point_A = (54,-15)
        point_B = (56,-12)
        transect = coast.Transect_t( nemo_t, point_A, point_B )
        or 
        transect = coast.Transect_t( nemo_t, y_indices=y_ind, x_indices=x_ind )
        
    Parameters
    ----------
    nemo_t : NEMO object  on the t-grid
    point_A : tuple, (lat,lon)
    point_B : tuple, (lat,lon)
    y_indices : 1d array of model y indices defining the points of the transect 
    x_indices : 1d array of model x indices defining the points of the transect 

    '''
    
    def __init__(self, nemo_t: COAsT, point_A: tuple=None, point_B: tuple=None,
                 y_indices=None, x_indices=None):
        super().__init__(nemo_t, point_A, point_B, y_indices, x_indices)
        
        
    def construct_pressure( self, ref_density=None, z_levels=None, extrapolate=False ):   
        '''
            This method is for calculating the hydrostatic and surface pressure fields
            on horizontal levels in the vertical (z-levels). The motivation 
            is to enable the calculation of horizontal gradients; however, 
            the variables can quite easily be interpolated onto the original 
            vertical grid.
             
            Requirements: The object's t-grid dataset must contain the sea surface height,
            Practical Salinity and the Potential Temperature variables.
            The GSW package is used to calculate the Absolute Pressure, 
            Absolute Salinity and Conservate Temperature.
            
            Three new variables (density, hydrostatic pressure, surface pressure)
            are created and added to the Transect_t.data dataset:
                density_zlevels       (t_dim, depth_z_levels, r_dim)
                pressure_h_zlevels    (t_dim, depth_z_levels, r_dim)
                pressure_s            (t_dim, r_dim)
            
            Note that density is constructed using the EOS10
            equation of state.

        Parameters
        ----------
        ref_density: float
            reference density value, if None, then the transect mean across time, 
            depth and along transect will be used.
        z_levels : (optional) numpy array
            1d array that defines the depths to interpolate the density and pressure
            on to. If not supplied, the Transect.gen_z_levels method will be used.
        extrapolate : boolean, default False
            If true the variables are extrapolated to the deepest level, if false,
            values below the bathymetry are set to NaN


        '''        
        
        # If there is no time dimension, add one, this is so
        # indexing can assume a time dimension exists
        if 't_dim' not in self.data.dims:
            self.data = self.data.expand_dims(dim={'t_dim':1},axis=0)

        # Generate vertical levels if not supplied
        if z_levels is None:   
            z_levels = Transect.gen_z_levels( self.data.bathymetry.max().item() )
        
        shape_ds = ( self.data.t_dim.size, len(z_levels), self.data.r_dim.size )
        salinity_z = np.ma.zeros( shape_ds )
        temperature_z = np.ma.zeros( shape_ds ) 
        salinity_s = self.data.salinity.to_masked_array()
        temperature_s = self.data.temperature.to_masked_array()
        s_levels = self.data.depth_0.values
        
        # Interpolate salinity and temperature onto z-levels
        # Note. At the current time there does not appear to be a good algorithm for 
        # performing this type of interpolation without loops, which can be a bottleneck.
        # Griddata is an option but does not support extrapolation and did not 
        # have noticable performance benefit.
        for it in self.data.t_dim:
            for ir in self.data.r_dim:
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
                        salinity_z[it,:,ir] = np.where( z_levels <= self.data.bathymetry.values[ir], 
                                sal_func(z_levels), np.nan )
                        temperature_z[it,:,ir] = np.where( z_levels <= self.data.bathymetry.values[ir], 
                                temp_func(z_levels), np.nan ) 
                    
        if extrapolate is False:
            # remove redundent levels    
            active_z_levels = np.count_nonzero(~np.isnan(salinity_z),axis=1).max() 
            salinity_z = salinity_z[:,:active_z_levels,:]
            temperature_z = temperature_z[:,:active_z_levels,:]
            z_levels = z_levels[:active_z_levels]
        
        # Absolute Pressure (depth must be negative)   
        pressure_absolute = np.ma.masked_invalid(
            gsw.p_from_z( -z_levels[:,np.newaxis], self.data.latitude ) )         
        # Absolute Salinity           
        salinity_absolute = np.ma.masked_invalid(
            gsw.SA_from_SP( salinity_z, pressure_absolute, self.data.longitude, 
            self.data.latitude ) )
        salinity_absolute = np.ma.masked_less(salinity_absolute,0)
        # Conservative Temperature
        temp_conservative = np.ma.masked_invalid(
            gsw.CT_from_pt( salinity_absolute, temperature_z ) )
        # In-situ density
        density_z = np.ma.masked_invalid( gsw.rho( 
            salinity_absolute, temp_conservative, pressure_absolute ) )
        
        coords={'depth_z_levels': (('depth_z_levels'), z_levels),
                'latitude': (('r_dim'), self.data.latitude),
                'longitude': (('r_dim'), self.data.longitude)}
        dims=['depth_z_levels', 'r_dim']
        attributes = {'units': 'kg / m^3', 'standard name': 'In-situ density on the z-level vertical grid'}
        
        if shape_ds[0] != 1:
            coords['time'] = (('t_dim'), self.data.time.values)
            dims.insert(0, 't_dim')
         
        if ref_density is None:    
            ref_density = np.mean( density_z )
        self.data['density_zlevels'] = xr.DataArray( np.squeeze(density_z), 
                coords=coords, dims=dims, attrs=attributes )
    
        # Cumulative integral of perturbation density on z levels
        density_cumulative = -cumtrapz( density_z - ref_density, x=-z_levels, axis=1, initial=0)
        hydrostatic_pressure = density_cumulative * self.GRAVITY
        
        attributes = {'units': 'kg m^{-1} s^{-2}', 'standard name': 'Hydrostatic perturbation pressure on the z-level vertical grid'}
        self.data['pressure_h_zlevels'] = xr.DataArray( np.squeeze(hydrostatic_pressure), 
                coords=coords, dims=dims, attrs=attributes )        
        self.data['pressure_s'] = ref_density * self.GRAVITY * self.data.ssh.squeeze()
        self.data.pressure_s.attrs = {'units': 'kg m^{-1} s^{-2}', 
                                  'standard_name': 'Surface perturbation pressure'}
        

