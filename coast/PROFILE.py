import numpy as np
import xarray as xr
from . import general_utils, plot_util, crps_util, COAsT
from . import crps_util as cu
import matplotlib.pyplot as plt
import glob
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import scipy.interpolate as interp

class PROFILE(COAsT):
    '''
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimensions:
        
        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a 
                     individual location.
        > z_dim   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files. There will be a fixed number of levels per dataset
                     despite the actual number of depth levels in a profile.
                     Empty depth levels for a given profile should be NaN.
                     
    Coordinates should be at least:
        > longitude (profile) :: Longitude of profile index
        > latitude (profile)  :: Latitude of profile index 
        > time (profile)      :: Time of profile index
        > depth (profile, z_dim) :: Depth at level index for profile index.
    
    Key variables (if present) should have names:
        > temperature (profile, z_dim)
        > potential_temperature (profile, z_dim)
        > practical_salinity (profile, z_dim)
        
    Other variables and coordinates can be included.
    '''
##############################################################################
###                ~ Initialisation and File Reading ~                     ###
##############################################################################

    def __init__(self):
        self.dataset = None
        return
    
    def read_EN4(self,fn_en4, multiple = False, chunks = {}):
        '''
        Reads a single or multiple EN4 netCDF files into the COAsT profile
        data structure.
        '''
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks = chunks)
        else:
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]
                
            file_to_read = []
            for file in fn_en4:
                if '*' in file:
                    wildcard_list = glob.glob(file)
                    file_to_read = file_to_read + wildcard_list
                else:
                    file_to_read.append(file)
                    
            # Reorder files to read 
            file_to_read = np.array(file_to_read)
            dates = [ff[-9:-3] for ff in file_to_read]
            dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]),1) for dd in dates]
            sort_ind = np.argsort(dates)
            file_to_read = file_to_read[sort_ind]
                    
            for ff in range(0,len(file_to_read)):
                file = file_to_read[ff]
                data_tmp = xr.open_dataset(file, chunks = chunks)
                if ff==0:
                    self.dataset = data_tmp
                else:
                    self.dataset=xr.concat((self.dataset, data_tmp),dim='N_PROF')
                
        
        rename_vars = {'LATITUDE':'latitude', 'LONGITUDE' : 'longitude',
                       'DEPH_CORRECTED' : 'depth', 'JULD':'time',
                       'POTM_CORRECTED':'potential_temperature',
                       'TEMP':'temperature', 'PSAL_CORRECTED':'practical_salinity',
                       'POTM_CORRECTED_QC':'qc_potential_temperature',
                       'PSAL_CORRECTED_QC':'qc_practical_salinity',
                       'DEPH_CORRECTED_QC':'qc_depth',
                       'JULD_QC':'qc_time',
                       'QC_FLAGS_PROFILES':'qc_flags_profiles',
                       'QC_FLAGS_LEVELS':'qc_flags_levels'}
        rename_dims = {'N_PROF':'profile', 'N_PARAM':'parameter', 
                       'N_LEVELS':'z_dim',}
        vars_to_keep = list(rename_vars.keys())
        coords = ['LATITUDE','LONGITUDE','JULD']
        self.dataset = self.dataset.set_coords(coords)
        self.dataset = self.dataset.rename(rename_dims)
        self.dataset = self.dataset[vars_to_keep].rename_vars(rename_vars)
        return
        
##############################################################################
###                ~            Manipulate           ~                     ###
##############################################################################
    
    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.
        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        ind = general_utils.subset_indices_lonlat_box(self.dataset.longitude, 
                                                      self.dataset.latitude, 
                                                      lonbounds[0], lonbounds[1],
                                                      latbounds[0], latbounds[1])
        return ind
    
##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################

    def plot_profile(self, var:str, profile_indices=None ):
        
        fig = plt.figure(figsize=(7,10))
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['profile'])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(profile=ii)
            prof_depth = self.dataset.depth.isel(profile=ii)
            ax = plt.plot(prof_var, prof_depth)
            
        plt.gca().invert_yaxis()
        plt.xlabel(var + '(' + self.dataset[var].units + ')')
        plt.ylabel('Depth (' + self.dataset.depth.units + ')')
        plt.grid()
        return fig, ax
    
    def plot_map(self, profile_indices=None, var_str=None, depth_index=None):
        
        if profile_indices is None:
            profile_indices=np.arange(0,self.dataset.dims['profile'])
        
        profiles=self.dataset.isel(profile=profile_indices)
        
        if var_str is None:
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values, s=5)
        else:
            print(profiles)
            c = profiles[var_str].isel(level=depth_index)
            fig, ax = plot_util.geo_scatter(profiles.longitude.values,
                                            profiles.latitude.values,
                                            c = c, s=5)
        
        return fig, ax
    
    def plot_ts_diagram(self, profile_index, var_t='potential_temperature', 
                        var_s='practical_salinity'):
        
        profile = self.dataset.isel(profile=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)
        
        return fig, ax

##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################

    def extract_profiles(self, nemo, fn_out=None, run_name = 'Undefined', 
                            z_interp = 'linear', nemo_frequency = 'daily',
                            en4_sal_name = 'practical_salinity',
                            en4_tem_name = 'potential_temperature',
                            v3p6 = False, fn_mesh_mask = None, 
                            bathy_name = 'hbatt'):
        '''
        VERSION 1.4 (05/07/2021)
        
        Extracts and does some basic analysis and identification of model data at obs
        locations, times and depths. Writes extracted data to file. This routine
        does the analysis per file, for example monthly files. Output can be
        subsequently input to analyse_ts_regional. Data is extracted saved in two
        forms: on the original model depth levels and interpolated onto the
        observed depth levels.
        
        This routine can be used in a loop to loop over all files containing
        daily or 25hourm data. Output files can be concatenated using a tools such
        as ncks or the concatenate_output_files() routine in this module.
        This is the recommended useage.
        
        INPUTS:
         nemo                 : NEMO object created on t-grid
         fn_out (str)         : Absolute filepath for desired output file.
                                If not provided, a delayed dataset will be returned
         run_name (str)       : Name of run. [default='Undefined']
                                Will be saved to output
         z_interp (str)       : Type of scipy interpolation to use for depth interpolation
                                [default = 'linear']
         nemo_frequency (str) : Time frequency of NEMO daily. String inputs can
                                be 'hourly', 'daily' or 'monthly'. Alternatively,
                                provide an integer number of hours.
         en4_sal_name (str)   : Name of EN4 (COAST) variable to use for salinity
                                [default = 'practical_salinity']
         en4_tem_name (str)   : Name of EN4 (COAST) variable to use for temperature
                                [default = 'potential_temperature']
         fn_mesh_mask (str)   : Full path to mesh_mask.nc if data is NEMO v3.6
                                
        OUTPUTS:
         Returns a new PROFILE object containing an uncomputed dataset and 
         can write to file (recommended):
         Writes extracted data to file. Extracted dataset has the dimensions:
             ex_level : Model level for directly extracted data
             level    : Observation level for interpolated model and EN4 data
             profile  : Profile location
         Output variables:
             mod_tem       : Model temperature interpolated onto obs depths
             obs_tem       : EN4 tempersture profiles
             mod_sal       : Model salinity interpolated onto obs depths
             obs_sal       : EN4 salinity profiles
             obs_z         : EN4 depths
             ex_mod_tem    : Model temperature profiles (directly extracted)
             ex_mod_sal    : Model salinity profiles (directly extracted)
             ex_depth      : Model depth at profiles (directly extracted)
             nn_ind_x      : Nearest neighbour x (column) indices
             nn_ind_y      : Nearest neighbour y (row) indices
             season        : Season indices -> 0 = DJF, 1=JJA, 2=MAM, 3=SON
             interp_dist   : Distances from nearest model cell to EN4 locations
             interp_lag    : Time lags between interpolated locations
             error_tem     : Model-EN4 temperature differences at observation depths
             error_sal     : Model-EN4 salinity differences at observation depths
             abs_error_tem : Model-EN4 abs. temperature differences at obs depths
             abs_error_sal : Model-EN4 abs. salinity differences at obs depths
             me_tem        : Mean temperature difference over each profile
             me_sal        : Mean salinity difference over each profile
             mae_tem       : Mean absolute temperature difference over each profile
             mae_sal       : Mean absolute salinity difference over each profile
        '''
        
        # 1) Sort out some NEMO variables first
        if ~v3p6:
            nemo.dataset['landmask'] = (('y_dim','x_dim'), nemo.dataset.bottom_level==0)
        else:
            dom = xr.open_dataset(fn_mesh_mask) 
            nemo.dataset['landmask'] = (('y_dim','x_dim'), dom.mbathy.squeeze()==0)
            nemo.dataset['bottom_level'] = (('y_dim', 'x_dim'), dom.mbathy.squeeze())
            nemo.dataset['bathymetry'] = (('y_dim', 'x_dim'), dom[bathy_name].squeeze())
    
        nemo = nemo.dataset[['temperature','salinity','depth_0', 'landmask','bathymetry', 'bottom_level']]
        nemo = nemo.rename({'temperature':'tem','salinity':'sal'})
        mod_time = nemo.time.values
            
        # 2) Read EN4, then extract desired variables
        en4 = self.dataset[[en4_tem_name, en4_sal_name,'depth',
                           'qc_flags_profiles', 'qc_flags_levels','time']]
        en4 = en4.rename({en4_sal_name:'sal', en4_tem_name:'tem'})
        
        # 4) Use only observations that are within model time window.
        en4_time = en4.time.values
        if nemo_frequency == 'hourly':
            nemo_frequency = 1
        elif nemo_frequency == 'daily':
            nemo_frequency = 24
        elif nemo_frequency == 'monthly':
            nemo_frequency = 30

        time_max = pd.to_datetime( max(mod_time) ) + relativedelta(hours=nemo_frequency/2)
        time_min = pd.to_datetime( min(mod_time) ) - relativedelta(hours=nemo_frequency/2)
            
        ind = np.logical_and( en4_time >= time_min, en4_time <= time_max )
        en4 = en4.isel(profile=ind)
        print('EN4 subsetted to model time period:', flush=True)
        print('    >>> {0} -> {1}'.format(str(time_min), str(time_max)), flush=True)
        print('  ', flush=True)
        
        # LOAD all remaining EN4 data
        en4.load()
        
        # SPATIAL indices
        ind2D = general_utils.nearest_indices_2D(nemo['longitude'], nemo['latitude'],
                                           en4['longitude'], en4['latitude'], 
                                           mask=nemo.landmask)
        
        print('Spatial Indices Calculated', flush=True)
        
        # TIME indices
        en4_time = en4.time.values
        ind_time = [ np.argmin( np.abs( mod_time - en4_time[tt] ) ) for tt in range(en4.dims['profile'])]
        ind_time = xr.DataArray(ind_time)
        print('Time Indices Calculated', flush=True)
        
        # INDEX the data and load
        mod_profiles = nemo.isel(x_dim=ind2D[0], y_dim=ind2D[1], t_dim=ind_time)
        mod_profiles = mod_profiles.rename({'dim_0':'profile'})
        mod_profiles['depth_0'] = mod_profiles['depth_0'].astype('float32')
        mod_profiles['longitude'] = mod_profiles['longitude'].astype('float32')
        mod_profiles['latitude'] = mod_profiles['latitude'].astype('float32')
        mod_profiles['bathymetry'] = mod_profiles['bathymetry'].astype('float32')
        print('Indexing model data. May take a while..')
        mod_profiles.load()
        print('Model indexed and loaded', flush=True)
        
        # Define variable arrays for interpolated data for monthly EN4 data
        n_mod_levels = mod_profiles.dims['z_dim']
        n_obs_levels = en4.dims['z_dim']
        n_prof = en4.dims['profile']
        data = xr.Dataset(coords = dict(
                              longitude=     (["profile"], en4.longitude.values.astype('float32')),
                              latitude=      (["profile"], en4.latitude.values.astype('float32')),
                              time=          (["profile"], en4.time.values),
                              level=         (['z_dim'], np.arange(0,n_obs_levels)),
                              ex_longitude = (["profile"], mod_profiles.longitude.values),
                              ex_latitude =  (["profile"], mod_profiles.latitude.values),
                              ex_time =      (["profile"], mod_profiles.time.values),
                              ex_level =     (["ex_z_dim"], np.arange(0, n_mod_levels))),
                          data_vars = dict(
                              mod_tem = (['profile','z_dim'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_tem = (['profile','z_dim'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              mod_sal = (['profile','z_dim'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_sal = (['profile','z_dim'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_z =    (['profile','z_dim'],    np.zeros((n_prof , n_obs_levels))*np.nan),
                              ex_mod_tem = (["profile", "ex_z_dim"], mod_profiles.tem.values),
                              ex_mod_sal = (["profile", "ex_z_dim"], mod_profiles.sal.values),
                              ex_depth =   (["profile", "ex_z_dim"], mod_profiles.depth_0.values.T),
                              nn_ind_x = (["profile"], ind2D[0]),
                              nn_ind_y = (["profile"], ind2D[1]),
                              bathy = (['profile'], mod_profiles.bathymetry.values),
                              original_dim_x = nemo.dims['x_dim'],
                              original_dim_y = nemo.dims['y_dim']))
        
        
        data['obs_tem'] = (['profile', 'z_dim'], en4.tem.values)
        data['obs_sal'] = (['profile','z_dim'], en4.tem.values)
        data['obs_z'] = (['profile', 'z_dim'], en4.depth.values)
        
        # Calculate interpolation distances
        interp_dist = general_utils.calculate_haversine_distance(
                                             en4.longitude, 
                                             en4.latitude, 
                                             mod_profiles.longitude, 
                                             mod_profiles.latitude)
        data['interp_dist'] = ('profile', interp_dist)
        
        # Calculate interpolation lags
        interp_lag = (data.ex_time.values - data.time.values).astype('timedelta64[h]')
        data['interp_lag'] = ('profile', interp_lag.astype(int))
        
        # Define the bottom level of observation data (to save processing every
        # level)
        obs_bl = [ max(np.where( ~np.isnan(en4.depth[pp]) )[0] ) for pp in range(en4.dims['profile'])]
        
        # Now loop over profiles and interpolate model onto obs.
        print('Interpolating onto observation depths...')
        for pp in range(0,n_prof):
            
            z_indices = np.arange(0,n_obs_levels)
            
            # Select the current profile
            mod_profile = mod_profiles.isel(profile = pp)
            obs_profile = en4.isel(profile = pp, z_dim=np.arange(0, obs_bl[pp]))
            z_indices = z_indices[:obs_bl[pp]]
            
            # Use bottom_level to mask dry depths
            if 'bottom_level' in mod_profile:
                bl = mod_profile.bottom_level.squeeze().values
                mod_profile = mod_profile.isel(z_dim=range(0,bl))
                    
            # Interpolate model to obs depths
            if ~all(np.isnan(obs_profile.tem.values)):
                
                # Interpolate model to obs levels
                f = interp.interp1d(mod_profile.depth_0.values, 
                                    mod_profile.tem.values, fill_value=np.nan, 
                                    bounds_error=False, kind = z_interp)
                data['mod_tem'][pp, :obs_bl[pp]] = np.float32( f( obs_profile.depth.values ) )
                
            if ~all(np.isnan(obs_profile.sal.values)):
                
                # Interpolate model to obs levels
                f = interp.interp1d(mod_profile.depth_0.values, 
                                    mod_profile.sal.values, fill_value=np.nan, 
                                    bounds_error=False, kind = z_interp)
                data['mod_sal'][pp, :obs_bl[pp]] = np.float32( f( obs_profile.depth.values ) )
            
            # Mask model data according to obs mask
            
        print(' Interpolated Profiles.', flush=True)
        
        # Define seasons as month numbers and identify datapoint seasons
        month_season_dict = {1:0, 2:0, 3:2, 4:2, 5:2, 6:1,
                             7:1, 8:1, 9:3, 10:3, 11:3, 12:0}
        pd_time = pd.to_datetime(data.time.values)
        pd_month = pd_time.month
        season_save = [month_season_dict[ii] for ii in pd_month]
        
        data['season'] = ('profile', season_save)
        data.attrs['run_name'] = run_name
        
        # Errors at all depths
        data["error_tem"] = (['profile','z_dim'], data.mod_tem - data.obs_tem)
        data["error_sal"] = (['profile','z_dim'], data.mod_sal - data.obs_sal)
        
        # Absolute errors at all depths
        data["abs_error_tem"] = (['profile','z_dim'], np.abs(data.error_tem))
        data["abs_error_sal"] = (['profile','z_dim'], np.abs(data.error_sal))
        
        # Mean errors across depths
        data['me_tem'] = ('profile', np.nanmean(data.error_tem, axis=1))
        data['me_sal'] = ('profile', np.nanmean(data.error_sal, axis=1))
        
        # Mean absolute errors across depths
        data['mae_tem'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
        data['mae_sal'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
                  
        # Write monthly stats to file
        if fn_out is not None:
            print('Writing File: {0}'.format(fn_out), flush=True)
            general_utils.write_ds_to_file(data, fn_out, mode='w', unlimited_dims='profile')
            
            print(' >>>>>>>  File Written.', flush=True)
            
        return_prof = PROFILE()
        return_prof.dataset = data
        return return_prof
        
    def analyse_profiles(self, ref_depth, fn_out=None,
                        ref_depth_method = 'interp',
                        regional_masks=[], region_names=[],
                        start_date = None, end_date = None, 
                        dist_omit=100, lag_omit=12):
        '''
        VERSION 1.4 (05/07/2021)
        
        Routine for doing REGIONAL and SEASONAL averaging of analysis files outputted using 
        extract_ts_per_file(). INPUT is the output from the analysis and OUTPUT is a file 
        containing REGIONAL averaged statistics. Multiple files can be provided
        to this analysis but it will be quicker if concatenated beforehand.
    
        INPUTS:
         fn_nemo_domain. (str)   : Absolute path to NEMO domain_cfg.nc or mesh_mask.nc
                                   Used only for bathymetry
         fn_extracted    (str)   : Absolute path to single analysis file
         fn_out          (str)   : Absolute path to desired output file
         ref_depth.      (array) : 1D array describing the reference depths onto which model
                                   and observations will be interpolated
         ref_depth_method (str)  : 'interp' or 'bin'. If interp, then routine will
                                   interpolate both model and observed values from
                                   observation depths onto the common ref_depth. If
                                   bin, then ref_depth will be treated as the 
                                   boundaries of averaging bins. BIN CURRENTLY
                                   UNIMPLEMENTED. [default = 'interp']
         regional_masks. (list)  : List of 2D bool arrays describing averaging regions. Each 
                                   array should have same shape as model domain. True 
                                   indicates a model point within the defined region. Will 
                                   always do a 'Whole Domain' region, even when undefined.
         region_names.   (list)  : List of strings. Names for each region in regional masks.
                                   To be saved in output file.
         start_date (datetime)   : Start date for analysis
         end_date (datetime)     : End date for analysis
         dist_omit (float)       : Distance of nearest grid cell from observation
                                   at which to omit the datapoint from averaging (km)
         lag_omit (float)        : Difference between model and obs time of
                                   nearest interpolation as which to reject from
                                   analysis (hours)
                                
        OUTPUTS
         Writes averages statistics to file. netCDF file has dimensions:
             profile   : Profile location dimension
             ref_depth : Reference depth dimension
                         If ref_depth_method == 'bin', then this will be bin
                         midpoints.
             region    : Regional dimension
             season    : Season dimension
         Data Variables:
             mod_tem   : Model temperature on reference depths/bins
             mod_sal   : Model salinity on reference depths/bins
             obs_tem   : Observed temperature on reference depths/bins
             obs_sal   : Observed salinity on reference depths/bins
             error_tem : Temperature errors on reference depths
             error_sal : Salinity errors on reference depths
             abs_error_tem : Absolute temperature err on reference depths
             abs_error_sal : Absolute salinity err on reference depths
             prof_mod_tem  : Regional/seasonal averaged model temperatures
             prof_mod_sal  : Regional/seasonal averaged model salinity
             prof_obs_tem  : Regional/seasonal averaged observed temperature
             prof_obs_sal  : Regional/seasonal averaged obs salinity
             prof_error_tem     : Regional/seasonal averaged temperature error
             prof_error_sal     : Regional/seasonal averaged salinity error
             prof_abs_error_tem : Regional/seasonal averaged abs. temp. error
             prof_abs_error_sal : Regional/seasonal averaged abs. sal. error
             mean_bathy   : Mean model bathymetric depths for profiles used in each
                            region/season
             is_in_region : Boolean array. Described whether each profile is within
                            each region
             start_date   : Start date for analysis
             end_date     : End date for analysis
        '''
    
        ds_ext = self.dataset
    
        # Cast inputs to numpy array
        ref_depth = np.array(ref_depth)
        
        # Make a copy of regional masks to avoid any memory leaking type loops
        regional_masks = regional_masks.copy()
        region_names = region_names.copy()
        
        # Append whole domain mask
        print('Defining Whole Domain region..')
        regional_masks.append(np.ones((int( ds_ext.original_dim_y.values ),
                                      int( ds_ext.original_dim_x.values ))))
        region_names.append('whole_domain')
        
        # Get numbers for array sizes
        n_regions = len(regional_masks)
        n_profiles = ds_ext.dims['profile']
        
        # Load only those variables that we want for interpolating to reference depths.
        
        ds = ds_ext[['mod_tem','obs_tem','mod_sal','obs_sal','obs_z', 'nn_ind_x', 'nn_ind_y', 'interp_dist',
                     'interp_lag', 'bathy']]
        print('Loading required data..')
        ds.load()
        
        # Restrict time if required or define start and end dates
        print('Constraining dates to {0} -> {1}'.format(start_date, end_date))
        print('(IF None then no constraint).')
        if start_date is not None:
            t_ind = pd.to_datetime( ds.time.values ) >= start_date
            ds = ds.isel(profile=t_ind)
        else:
            start_date = min(ds.time)
            
        if end_date is not None:
            t_ind = pd.to_datetime( ds.time.values ) <= end_date
            ds = ds.isel(profile=t_ind)
        else:
            end_date = min(ds.time)
       
        # Update number of profiles
        n_profiles = ds.dims['profile']
        
        # Figure out which points lie in which region
        print('Figuring out which regions each profile is in..')
        is_in_region = [mm[ds.nn_ind_y.values.astype(int), ds.nn_ind_x.values.astype(int)] for mm in regional_masks]
        is_in_region = np.array(is_in_region, dtype=bool)
        
        #Figure out ref depths or bins to create interp array
        if ref_depth_method == 'interp':
            n_ref_depth = len(ref_depth)
        if ref_depth_method == 'bin':
            n_ref_depth = len(ref_depth) - 1
            bin_widths = np.array( [ref_depth[ii+1] - ref_depth[ii] for ii in np.arange(0,n_ref_depth)] )
            bin_mids = np.array( [ref_depth[ii] + .5*ref_depth[ii] for ii in np.arange(0,n_ref_depth)] )
        
        # Create dataset for interpolation
        ds_interp = xr.Dataset(coords = dict(
                                   ref_depth = ('ref_depth', ref_depth),
                                   longitude = ('profile', ds.longitude.values),
                                   latitude = ('profile', ds.latitude.values),
                                   time = ('profile', ds.time.values),
                                   region = ('region', region_names)),
                               data_vars = dict(
                                   bathy = ('profile', ds.bathy.values),
                                   mod_tem = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                                   mod_sal = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                                   obs_tem = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan),
                                   obs_sal = (['profile','ref_depth'], np.zeros((n_profiles, n_ref_depth), dtype='float32')*np.nan)))
        
        # INTERP1 = interpolate the obs and model to reference depths from the
        # OBS depths. Model already interpolated in extract routine.
        print('Interpolate both model and obs onto reference depths.')
        if ref_depth_method=='interp':
            for pp in range(0, n_profiles):
                prof = ds.isel(profile=pp)
                
                mask = np.isnan( prof.obs_sal )
                if ~all(np.isnan(mask)):
                    prof['mod_sal'][mask] = np.nan
                    f = interp.interp1d(prof.obs_z.values, prof.mod_sal.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['mod_sal'][pp] = f( ref_depth )
                    f = interp.interp1d(prof.obs_z.values, prof.obs_sal.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['obs_sal'][pp] = f( ref_depth )
            
                mask = np.isnan( prof.obs_tem )
                if ~all(np.isnan(mask)):
                    prof['mod_tem'][mask] = np.nan
                    f = interp.interp1d(prof.obs_z.values, prof.mod_tem.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['mod_tem'][pp] = f( ref_depth )
                    f = interp.interp1d(prof.obs_z.values, prof.obs_tem.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['obs_tem'][pp] = f( ref_depth )
                    
        # BIN = Bin into depth bins rather than interpolate - NOT USED CURRENTLY
        elif ref_depth_method=='bin':
            raise NotImplementedError()
            
        # Calculate errors with depth
        print('Calculating errors..')
        ds_interp['error_tem'] = (ds_interp.mod_tem - ds_interp.obs_tem).astype('float32')
        ds_interp['error_sal'] = (ds_interp.mod_sal - ds_interp.obs_sal).astype('float32')
        ds_interp['abs_error_tem'] = np.abs( (ds_interp.mod_tem - ds_interp.obs_tem).astype('float32') )
        ds_interp['abs_error_sal'] = np.abs( (ds_interp.mod_sal - ds_interp.obs_sal).astype('float32') )
        
        # Define dataset for regional averaging
        empty_array = np.zeros((n_regions, 5, n_ref_depth), dtype='float32')*np.nan
        ds_reg_prof = xr.Dataset(coords = dict(
                                    region = ('region',region_names),
                                    ref_depth = ('ref_depth', ref_depth),
                                    season = ('season', ['DJF','JJA','MAM','SON','All'])),
                                 data_vars = dict(
                                    prof_mod_tem = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_mod_sal = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_obs_tem = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_obs_sal = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_error_tem = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_error_sal = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_abs_error_tem = (['region','season','ref_depth'], empty_array.copy()),
                                    prof_abs_error_sal = (['region','season','ref_depth'], empty_array.copy()),
                                    mean_bathy = (['region','season'], np.zeros((n_regions, 5))*np.nan)))
        
        season_str_dict = {'DJF':0,'JJA':1,'MAM':2,'SON':3, 'All':4}
        
        # Remove flagged points
        print('Removing points further than {0}km'.format(dist_omit))
        print('Removing points further apart than {0} hours'.format(lag_omit))
        omit1 = ds.interp_dist.values <= dist_omit
        omit2 = ds.interp_lag.values <= lag_omit
        omit_flag = np.logical_and( omit1, omit2)
        ds_interp_clean = ds_interp.isel(profile = omit_flag)
        is_in_region_clean = is_in_region[:, omit_flag]
    
        # Loop over regional arrays. Assign mean to region and seasonal means
        print('Calculating regional and seasonal averages..')
        for reg in range(0,n_regions):
        	# Do regional average for the correct seasons
            reg_ind = np.where( is_in_region_clean[reg].astype(bool) )[0]
            if len(reg_ind)<1:
                continue
            reg_tmp = ds_interp_clean.isel(profile = reg_ind)
            reg_tmp_group = reg_tmp.groupby('time.season')
            reg_tmp_mean = reg_tmp_group.median(dim='profile', skipna=True).compute()
            season_str = reg_tmp_mean.season.values
            season_ind = [season_str_dict[ss] for ss in season_str]
            
            ds_reg_prof['prof_mod_tem'][reg, season_ind] = reg_tmp_mean.mod_tem
            ds_reg_prof['prof_mod_sal'][reg, season_ind] = reg_tmp_mean.mod_sal
            ds_reg_prof['prof_obs_tem'][reg, season_ind] = reg_tmp_mean.obs_tem
            ds_reg_prof['prof_obs_sal'][reg, season_ind] = reg_tmp_mean.obs_sal
            ds_reg_prof['prof_error_tem'][reg, season_ind] = reg_tmp_mean.error_tem
            ds_reg_prof['prof_error_sal'][reg, season_ind] = reg_tmp_mean.error_sal
            ds_reg_prof['prof_abs_error_tem'][reg, season_ind] = reg_tmp_mean.abs_error_tem
            ds_reg_prof['prof_abs_error_sal'][reg, season_ind] = reg_tmp_mean.abs_error_sal
            ds_reg_prof['mean_bathy'][reg, season_ind] = reg_tmp_mean.bathy
            
            # Do regional averaging across all seasons
            reg_tmp_mean = reg_tmp.median(dim='profile', skipna=True).compute()
            
            ds_reg_prof['prof_mod_tem'][reg, 4] = reg_tmp_mean.mod_tem
            ds_reg_prof['prof_mod_sal'][reg, 4] = reg_tmp_mean.mod_sal
            ds_reg_prof['prof_obs_tem'][reg, 4] = reg_tmp_mean.obs_tem
            ds_reg_prof['prof_obs_sal'][reg, 4] = reg_tmp_mean.obs_sal
            ds_reg_prof['prof_error_tem'][reg, 4] = reg_tmp_mean.error_tem
            ds_reg_prof['prof_error_sal'][reg, 4] = reg_tmp_mean.error_sal
            ds_reg_prof['prof_abs_error_tem'][reg, 4] = reg_tmp_mean.abs_error_tem
            ds_reg_prof['prof_abs_error_sal'][reg, 4] = reg_tmp_mean.abs_error_sal
            ds_reg_prof['mean_bathy'][reg, 4] = reg_tmp_mean.bathy
        
        # Drop bathy for some reason
        ds_interp = ds_interp.drop('bathy')
        
        # Merge output dataset
        ds_interp = xr.merge((ds_interp, ds_reg_prof))
        ds_interp['is_in_region'] = (['region','profile'], is_in_region)
        
        ds_interp['start_date'] = start_date
        ds_interp['end_date'] = end_date
        
        # Write output to file
        if fn_out is not None:
            print('Writing to file..')
            general_utils.write_ds_to_file(ds_interp, fn_out)
            print(' >>>>>>>  File Written: ' + fn_out, flush=True)
        ds_interp = ds_interp.drop(['ex_longitude', 'ex_latitude', 'ex_time'])
        return_prof = PROFILE()
        return_prof.dataset = ds_interp
        return return_prof
    
    def extract_top_and_bottom(self, nemo, fn_out=None, 
                 surface_def=2, bottom_def=10, crps_radii = [2,4,6],
                 nemo_frequency = 'hourly',
                 start_date = None, end_date = None,
                 en4_tem_name = 'potential_temperature',
                 en4_sal_name = 'practical_salinity',
                 v3p6 = False, fn_mesh_mask = None, 
                 bathy_name = 'hbatt'):
                 
        ''' 
        Extraction and analysis of model temperature and salinity 
        output with EN4 profiles. This routine should be called using a 
        preprocessed PROFILE object.
        
        Surface and bottom values will be estimated from the profile data
        using the surface_def and bottom_def variables. These are the 
        depths over which averages are taken to obtain estimates of SST, SSS,
        SBS and SBT. 
        
        Errors and mean absolute errors will be taken. At the surface, CRPS
        values will be calculated at provided radii.
        
        INPUT
         nemo (coast.NEMO)    : NEMO object containing t-grid data
         fn_out (str)         : Full path to output file. 
                                If None [default], then no file is saved
         surface_def (float)  : Definition of the 'surface' in metres - for averaging
         bottom_def (float)   : Definition of the 'bottom' in metres - for averaging
         crps_radii (array)   : Array/list of CRPS radii in number of grid cells
         nemo_frequency (str) : Frequency of NEMO data. Can be string ('daily',
                                'hourly', 'monthly') or a integer number of hours.
         start_date (datetime): Start date for analysis.
         end_date (datetime)  : End date for analysis.
         en4_tem_name (str)   : Name of EN4 (COAsT name) to use for temperature
         en4_sal_name (str)   : Name of EN4 (COAsT name) to use for salinity
         v3p6 (bool)          : Set True is data is NEMO v3.6
         fn_mesh_mask (str)   : If v3p6 == True, provide a full path to mesh_mask.nc
         bathy_name (str)     : If v3p6 == True, provide name of bathymetry in mesh_mask.nc
             
        OUTPUT
         New PROFILE() object containing a dataset of extracted data and
         averaged surface/bottom observations. This contains:
        '''
        
        ds = nemo.dataset
        
        # Open NEMO data files and define some arrays
        if ~v3p6:
            nemo_mask = ds.bottom_level == 0
        else:
            dom = xr.open_dataset(fn_mesh_mask)
            nemo_mask = dom.mbathy.squeeze() == 0
            dom.close()
            
        ds = ds.rename({'t_dim':'time'})
        ds = ds[['temperature_top','salinity_top','temperature_bot','salinity_bot',
                 'bathymetry']]
        
        # Restrict time if required or define start and end dates
        print('Constraining dates to {0} -> {1}'.format(start_date, end_date))
        print('(IF None then no constraint).')
        ds.time.load()
        if start_date is not None:
            t_ind = pd.to_datetime( ds.time.values ) >= start_date
            ds = ds.isel(time=t_ind)
        else:
            start_date = min(ds.time)
            
        if end_date is not None:
            t_ind = pd.to_datetime( ds.time.values ) <= end_date
            ds = ds.isel(time=t_ind)
        else:
            end_date = min(ds.time)
            
        # Cut out obs inside model time window
        if nemo_frequency == 'hourly':
            nemo_frequency = 1
        elif nemo_frequency == 'daily':
            nemo_frequency = 24
        elif nemo_frequency == 'monthly':
            nemo_frequency = 30*24
        
        en4 = self.dataset
        n_nemo_time = ds.dims['time']
        en4_time = en4.time.values
        mod_time = pd.to_datetime(ds.time.values)
        time_max = pd.to_datetime( max(mod_time) ) + relativedelta(hours=nemo_frequency/2)
        time_min = pd.to_datetime( min(mod_time) ) - relativedelta(hours=nemo_frequency/2)
        ind = np.logical_and( en4_time >= time_min, en4_time <= time_max )
        en4 = en4.isel(profile=ind)
        print('EN4 subsetted to model time period:', flush=True)
        print('    >>> {0} -> {1}'.format(str(time_min), str(time_max)), flush=True)
        print('  ', flush=True)
        
        # Save dimensions of dataset
        n_c = ds.dims['x_dim']
        n_r = ds.dims['y_dim']
        
        # Get nearest model indices to observations
        en4_time = pd.to_datetime(en4.time.values)
        n_prof = len(en4_time)
        ind2D = general_utils.nearest_indices_2D(ds.longitude, ds.latitude, 
                                      en4.longitude, en4.latitude,
                                      mask=nemo_mask)
        
        # Determine interp_dist
        mod_lon = ds.longitude.isel(x_dim = ind2D[0], y_dim=ind2D[1]).values
        mod_lat = ds.latitude.isel(x_dim=ind2D[0], y_dim=ind2D[1]).values
        obs_lon = en4.longitude.values
        obs_lat = en4.latitude.values
        interp_dist = general_utils.calculate_haversine_distance( mod_lon, mod_lat, 
                                                       obs_lon, obs_lat )
        print('Calculated nearest model indices.', flush=True)
        
        # Estimate EN4 SST as mean of top levels
        print('Averaging EN4 over top {0}m for surface definition'.format(surface_def), flush=True)
        surface_ind = en4.depth <= surface_def
        
        sst_en4 = en4[en4_tem_name].where(surface_ind, np.nan)
        sss_en4 = en4[en4_sal_name].where(surface_ind, np.nan)
        
        sst_en4 = sst_en4.mean(dim="z_dim", skipna=True).load()
        sss_en4 = sss_en4.mean(dim="z_dim", skipna=True).load()
        
        # Bottom values
        print('Averaging EN4 over bottom {0}m for bottom definition'.format(bottom_def), flush=True)
        bathy_pts = ds.bathymetry.isel(x_dim = ind2D[0], y_dim = ind2D[1]).swap_dims({'dim_0':'profile'})
        bottom_ind = en4.depth >= (bathy_pts - bottom_def)

        sbt_en4 = en4[en4_tem_name].where(bottom_ind, np.nan)
        sbs_en4 = en4[en4_sal_name].where(bottom_ind, np.nan)
    
        sbt_en4 = sbt_en4.mean(dim="z_dim", skipna=True).load()
        sbs_en4 = sbs_en4.mean(dim="z_dim", skipna=True).load()
        
        # Define analysis arrays
        n_prof = en4.dims['profile']
        n_crps = len(crps_radii)
        
        x_dim_len = ds.dims['x_dim']
        y_dim_len = ds.dims['y_dim']
        en4_season = general_utils.get_season_index(sst_en4.time.values)
        
        analysis = xr.Dataset(coords = dict(
                    longitude = ("profile", sst_en4.longitude.values),
                    latitude = ("profile", sst_en4.latitude.values),
                    time = ("profile", sst_en4.time.values),
                    season_ind = ("profile", en4_season),
                    crps_radius = ("crps_radius", crps_radii)),
                data_vars = dict(
                    obs_sst = ('profile', sst_en4.values),
                    obs_sss = ('profile', sss_en4.values),
                    obs_sbt = ("profile", sbt_en4.values),
                    obs_sbs = ("profile", sbs_en4.values),
                    mod_sst = ("profile", np.zeros(n_prof)*np.nan),
                    mod_sss = ("profile", np.zeros(n_prof)*np.nan),
                    mod_sbt = ("profile", np.zeros(n_prof)*np.nan),
                    mod_sbs = ("profile", np.zeros(n_prof)*np.nan),
                    sst_err = ("profile", np.zeros(n_prof)*np.nan),
                    sss_err = ("profile", np.zeros(n_prof)*np.nan),
                    sbt_err = ("profile", np.zeros(n_prof)*np.nan),
                    sbs_err = ("profile", np.zeros(n_prof)*np.nan),
                    sst_crps = (["profile","crps_radius"], np.zeros((n_prof, n_crps))*np.nan),
                    sss_crps = (["profile","crps_radius"], np.zeros((n_prof, n_crps))*np.nan),
                    nn_ind_x = ("profile", ind2D[0]),
                    nn_ind_y = ("profile", ind2D[1]),
                    interp_dist = ("profile", interp_dist),
                    original_dim_x = n_c,
                    original_dim_y = n_r))
        
        print('Starting analysis')
        
        # LOOP over model time snapshots -- For each snapshot identify all 
        # corresponding EN4 datapoints
        print('Looping over model time snapshots', flush=True)
        for tii in range(0, n_nemo_time):
            
            time_diff = np.abs( mod_time[tii] - en4_time ).astype('timedelta64[h]')
            use_ind = np.where( time_diff.astype(int) < nemo_frequency )[0]
            n_use = len(use_ind)
            
            if n_use>0:
                
                # Index the model data to time slice and load it.
                tmp = ds.isel(time = tii)
                tmp.load()
                x_tmp = ind2D[0][use_ind]
                y_tmp = ind2D[1][use_ind]
                
                x_tmp = xr.where(x_tmp<x_dim_len-7, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp<y_dim_len-7, y_tmp, np.nan)

                x_tmp = xr.where(x_tmp>7, x_tmp, np.nan)
                y_tmp = xr.where(y_tmp>7, y_tmp, np.nan)
                
                shared_mask = np.logical_or(np.isnan(x_tmp), np.isnan(y_tmp))
                shared_mask = np.where(~shared_mask)
                
                x_tmp = x_tmp[shared_mask].astype(int)
                y_tmp = y_tmp[shared_mask].astype(int)
                use_ind = use_ind[shared_mask].astype(int)
                
                n_use = len(use_ind)
                if n_use<1:
                    continue
                
                # Surface errors
                tmp_pts = tmp.isel(x_dim = x_tmp, y_dim = y_tmp)
                sst_en4_tmp = sst_en4.values[use_ind]
                sss_en4_tmp = sss_en4.values[use_ind]
                analysis['sst_err'][use_ind] = tmp_pts.temperature_top.values - sst_en4_tmp
                analysis['sss_err'][use_ind] = tmp_pts.salinity_top.values - sss_en4_tmp
                analysis['mod_sst'][use_ind] = tmp_pts.temperature_top.values
                analysis['mod_sss'][use_ind] = tmp_pts.salinity_top.values
                
                # Bottom errors
                sbt_en4_tmp = sbt_en4.values[use_ind]
                sbs_en4_tmp = sbs_en4.values[use_ind]
                analysis['sbt_err'][use_ind] = tmp_pts.temperature_bot.values - sbt_en4_tmp
                analysis['sbs_err'][use_ind] = tmp_pts.salinity_bot.values - sbs_en4_tmp
                analysis['mod_sbt'][use_ind] = tmp_pts.temperature_bot.values
                analysis['mod_sbs'][use_ind] = tmp_pts.salinity_bot.values
                
                # CRPS
                for cc in range(n_crps):
                    cr = crps_radii[cc]
                    nh_x = [np.arange( x_tmp[ii]-cr, x_tmp[ii]+(cr+1) ) for ii in range(0,n_use)] 
                    nh_y = [np.arange( y_tmp[ii]-cr, y_tmp[ii]+(cr+1) ) for ii in range(0,n_use)]   
                    nh = [tmp.isel(x_dim = nh_x[ii], y_dim = nh_y[ii]) for ii in range(0,n_use)] 
                    crps_tem_tmp = [ cu.crps_empirical(nh[ii].temperature_top.values.flatten(), sst_en4_tmp[ii]) for ii in range(0,n_use)]
                    crps_sal_tmp = [ cu.crps_empirical(nh[ii].salinity_top.values.flatten(), sss_en4_tmp[ii]) for ii in range(0,n_use)]
                    analysis['sst_crps'][use_ind, cc] = crps_tem_tmp
                    analysis['sss_crps'][use_ind, cc] = crps_sal_tmp
                    
        print('Profile analysis done', flush=True)
        
        # Define absolute errors
        analysis['sst_abs_err'] = (['profile'], np.abs(analysis['sst_err']))
        analysis['sss_abs_err'] = (['profile'], np.abs(analysis['sss_err']))
        analysis['sbt_abs_err'] = (['profile'], np.abs(analysis['sbt_err']))
        analysis['sbs_abs_err'] = (['profile'], np.abs(analysis['sbs_err']))                                 
            
        analysis['start_date'] = start_date
        analysis['end_date'] = end_date
        
        # Write to file
        if fn_out is not None:
            print('Writing File: {0}'.format(fn_out), flush=True)
            general_utils.write_ds_to_file(analysis, fn_out)
            
        return_prof = PROFILE()
        return_prof.dataset = analysis
        return return_prof
    
    
    def analyse_top_and_bottom(self, fn_out=None, 
                       regional_masks=[], region_names=[],
                       start_date = None, end_date = None,
                       dist_omit = 5):
        '''
        A routine for averaging the hourly analysis into regional subdomains.
        This routine will also average into seasons as well as over all time.
        The resulting output dataset will have dimension (region, season)
        
        INPUTS
         fn_out (str)          : Absolute path to desired output file
         regional_masks (list) : List of 2D boolean arrays. Where true, this is the region
                                 used for averaging. The whole domain will be added to the 
                                 list, or will be the only domain if left to be default.
         region_names (list)   : List of strings, the names used for each region.
         start_date (datetime) : Start date for analysis 
         end_date (datetime)   : End data for analysis
         dist_omit (float)     : Nearest neighbour distance at which to reject
                                 datapoint from analysis.
        '''
        
        ds_stats = self.dataset[['sst_err', 'sss_err', 'sbt_err', 'sbs_err',
                             'sst_crps', 'sss_crps', 'nn_ind_x', 'nn_ind_y',
                             'sst_abs_err', 'sss_abs_err', 'sbt_abs_err',
                             'sbs_abs_err', 'original_dim_x',
                             'original_dim_y', 'interp_dist']].load()
        
        # Restrict time if required or define start and end dates
        if start_date is not None:
            t_ind = pd.to_datetime( ds_stats.time.values ) >= start_date
            ds_stats = ds_stats.isel(profile=t_ind)
        else:
            start_date = min(ds_stats.time)
            
        if end_date is not None:
            t_ind = pd.to_datetime( ds_stats.time.values ) <= end_date
            ds_stats = ds_stats.isel(profile=t_ind)
        else:
            end_date = min(ds_stats.time)
            
        # Make a copy of regional masks to avoid any memory leaking type loops
        regional_masks = regional_masks.copy()
        region_names = region_names.copy()
        
        # Append whole domain mask
        print('Defining Whole Domain region..')
        regional_masks.append(np.ones((int( ds_stats.original_dim_y.values ),
                                      int( ds_stats.original_dim_x.values ))))
        region_names.append('whole_domain')
        n_regions = len(regional_masks)
        
        # Determine which region each point lies in
        is_in_region = [mm[ds_stats.nn_ind_y, ds_stats.nn_ind_x] for mm in regional_masks]
        is_in_region = np.array(is_in_region, dtype=bool)
        
        # 5 Seasons, define array
        reg_array = np.zeros((n_regions, 5))*np.nan
        n_crps = ds_stats.dims['crps_radius']
        
        # Define array to contain averaged data
        ds_mean = xr.Dataset(coords = dict(
                            longitude = ("profile", ds_stats.longitude.values),
                            latitude = ("profile", ds_stats.latitude.values),
                            time = ("profile", ds_stats.time.values),
                            region_names = ('region', region_names)),
                        data_vars = dict(
                            sst_me = (["region", "season"],  reg_array.copy()),
                            sss_me = (["region", "season"],  reg_array.copy()),
                            sst_mae = (["region", "season"], reg_array.copy()),
                            sss_mae = (["region", "season"], reg_array.copy()),
                            sst_estd = (["region", "season"],  reg_array.copy()),
                            sss_estd = (["region", "season"],  reg_array.copy()),
                            sbt_me = (['region','season'], reg_array.copy()),
                            sbs_me = (['region','season'], reg_array.copy()),
                            sbt_estd = (['region','season'], reg_array.copy()),
                            sbs_estd = (['region','season'], reg_array.copy()),
                            sbt_mae = (['region','season'], reg_array.copy()),
                            sbs_mae = (['region','season'], reg_array.copy()), 
                            sst_crps_mean = (["region", "season", "crps_radius"], np.zeros((n_regions, 5, n_crps))*np.nan),
                            sss_crps_mean = (["region", "season", "crps_radius"], np.zeros((n_regions, 5, n_crps))*np.nan)))
        
        season_indices = {'DJF':0, 'JJA':1, 'MAM':2, 'SON':3}
     
        # Loop over regions. For each, group into seasons and average.
        # Place into ds_mean dataset.
        
        print('Removing points further than {0}km'.format(dist_omit))
        omit_flag = ds_stats.interp_dist.values <= dist_omit
        ds_stats_clean = ds_stats.isel(profile = omit_flag)
        is_in_region_clean = is_in_region[:, omit_flag]
        
        for reg in range(0,n_regions):
            reg_ind = np.where( is_in_region_clean[reg].astype(bool) )[0]
            
            if len(reg_ind)<1:
                continue
    
            ds_reg = ds_stats_clean.isel(profile = reg_ind)
            ds_reg_group = ds_reg.groupby('time.season')
            
            # MEANS
            ds_reg_mean = ds_reg_group.mean(dim = 'profile', skipna=True).compute()
            
            s_in_mean = ds_reg_mean.season.values
            s_ind = np.array([season_indices[ss] for ss in s_in_mean], dtype=int)
       
            ds_mean['sst_me'][reg, s_ind]  = ds_reg_mean.sst_err.values
            ds_mean['sss_me'][reg, s_ind]  = ds_reg_mean.sss_err.values
            ds_mean['sst_mae'][reg, s_ind] = ds_reg_mean.sst_abs_err.values
            ds_mean['sss_mae'][reg, s_ind] = ds_reg_mean.sss_abs_err.values
            ds_mean['sst_crps_mean'][reg, s_ind] = ds_reg_mean.sst_crps.values
            ds_mean['sss_crps_mean'][reg, s_ind] = ds_reg_mean.sss_crps.values
            
            ds_mean['sbt_me'][reg, s_ind]  = ds_reg_mean.sbt_err.values
            ds_mean['sbs_me'][reg, s_ind]  = ds_reg_mean.sbs_err.values
            ds_mean['sbt_mae'][reg, s_ind] = ds_reg_mean.sbt_abs_err.values
            ds_mean['sbs_mae'][reg, s_ind] = ds_reg_mean.sbs_abs_err.values
            
            ds_reg_mean = ds_reg.mean(dim='profile', skipna=True).compute()
            ds_mean['sst_me'][reg, 4]  = ds_reg_mean.sst_err.values
            ds_mean['sss_me'][reg, 4]  = ds_reg_mean.sss_err.values
            ds_mean['sst_mae'][reg, 4] = ds_reg_mean.sst_abs_err.values
            ds_mean['sss_mae'][reg, 4] = ds_reg_mean.sss_abs_err.values
            ds_mean['sst_crps_mean'][reg, 4] = ds_reg_mean.sst_crps.values
            ds_mean['sss_crps_mean'][reg, 4] = ds_reg_mean.sss_crps.values
            
            ds_mean['sbt_me'][reg, 4]  = ds_reg_mean.sbt_err.values
            ds_mean['sbs_me'][reg, 4]  = ds_reg_mean.sbs_err.values
            ds_mean['sbt_mae'][reg, 4] = ds_reg_mean.sbt_abs_err.values
            ds_mean['sbs_mae'][reg, 4] = ds_reg_mean.sbs_abs_err.values
    
            # STD DEVIATIONS
            ds_reg_std = ds_reg_group.std(dim = 'profile', skipna=True).compute()
    
            s_in_std = ds_reg_std.season.values
            s_ind = np.array([season_indices[ss] for ss in s_in_std], dtype=int)
            
            ds_mean['sst_estd'][reg, s_ind]  = ds_reg_std.sst_err.values
            ds_mean['sss_estd'][reg, s_ind]  = ds_reg_std.sss_err.values
    
            ds_mean['sbt_estd'][reg, s_ind]  = ds_reg_std.sbt_err.values
            ds_mean['sbs_estd'][reg, s_ind]  = ds_reg_std.sbs_err.values
    
            ds_reg_std = ds_reg.std(dim='profile', skipna=True).compute()
            ds_mean['sst_estd'][reg, 4]  = ds_reg_std.sst_err.values
            ds_mean['sss_estd'][reg, 4]  = ds_reg_std.sss_err.values
    
            ds_mean['sbt_estd'][reg, 4]  = ds_reg_std.sbt_err.values
            ds_mean['sbs_estd'][reg, 4]  = ds_reg_std.sbs_err.values
        
        ds_mean['start_date'] = start_date
        ds_mean['end_date'] = end_date
        ds_mean['is_in_region'] = (['region', 'profile'], is_in_region)
        
        # Write to file    
        if fn_out is not None:
            print('Writing File: {0}'.format(fn_out), flush=True)
            general_utils.write_ds_to_file(ds_mean, fn_out, mode='w', 
                                           unlimited_dims='profile')
            
            print(' >>>>>>>  File Written. ', flush=True)
            
        return_prof = PROFILE()
        return_prof.dataset = ds_mean
        return return_prof
    
        
    def process_en4(self, fn_out=None, lonbounds=None, latbounds=None):
        '''
        VERSION 1.4 (05/07/2021)
        
        PREPROCESSES EN4 data ready for comparison with model data.
        This routine will cut out a desired geographical box of EN4 data and
        then apply quality control according to the available flags in the
        netCDF files. Quality control happens in two steps:
            1. Where a whole data profile is flagged, it is completely removed
               from the dataset
            2. Where a single datapoint is rejected in either temperature or 
               salinity, it is set to NaN.
        This routine attempts to use xarray/dask chunking magic to keep 
        memory useage low however some memory is still needed for loading
        flags etc. May be slow if using large EN4 datasets.
        
        Routine will return a processed profile object dataset and can write 
        the new dataset to file if fn_out is defined. If saving to the
        PROFILE object, be aware that DASK computations will not have happened
        and will need to be done using .load(), .compute() or similar before
        accessing the values. IF using multiple EN4 files or large dataset,
        make sure you have chunked the data over N_PROF dimension.
        
        INPUTS
         fn_out (str)      : Full path to a desired output file. If unspecified
                             then nothing is written.
         lonbounds (array) : Array of longitude bounds for cut out [lonmax, lonmin]
         latbounds (array) : Array of latitude bounds for cut out [latmax, latmin]
         
        EXAMPLE USEAGE:
         profile = coast.PROFILE()
         profile.read_EN4(fn_en4, chunks={'N_PROF':10000})
         fn_out = '~/output_file.nc'
         new_profile = profile.preprocess_en4(fn_out = fn_out, 
                                              lonbounds = [-10, 10], 
                                              latbounds = [45, 65])
        '''
        
        ds = self.dataset
        
        if lonbounds is not None and latbounds is not None:
            ind = general_utils.subset_indices_lonlat_box(ds['longitude'], 
                                                          ds['latitude'],
                                                          lonbounds[0], lonbounds[1], 
                                                          latbounds[0], latbounds[1])[0]
            ds = ds.isel(profile=ind)
        print('EN4 subsetted to model domain: ', flush=True)
        print('    >>> LON {0} -> {1}'.format(str(lonbounds[0]), str(lonbounds[1])), flush=True)
        print('    >>> LAT {0} -> {1}'.format(str(latbounds[0]), str(latbounds[1])), flush=True)
        
        # REJECT profiles that are QC flagged.
        print(' Applying QUALITY CONTROL to EN4 data...', flush=True)
        ds.qc_flags_profiles.load()
        
        # This line reads converts the QC integer to a binary string.
        # Each bit of this string is a different QC flag. Which flag is which can
        # be found on the EN4 website:
        # https://www.metoffice.gov.uk/hadobs/en4/en4-0-2-profile-file-format.html
        qc_str = [np.binary_repr(ds.qc_flags_profiles.values[pp]).zfill(30)[::-1] for pp in range(ds.dims['profile'])]
        
        # Determine indices of kept profiles
        reject_tem_prof = np.array( [int( qq[0] ) for qq in qc_str], dtype=bool )
        reject_sal_prof = np.array( [int( qq[1] ) for qq in qc_str], dtype=bool )
        reject_both_prof = np.logical_and( reject_tem_prof, reject_sal_prof )
        ds['reject_tem_prof'] = (['profile'], reject_tem_prof)
        ds['reject_sal_prof'] = (['profile'], reject_sal_prof)
        print('     >>> QC: Completely rejecting {0} / {1} profiles'.format(np.sum(reject_both_prof), ds.dims['profile']), flush=True)
        
        ds = ds.isel(profile=~reject_both_prof)
        reject_tem_prof = reject_tem_prof[~reject_both_prof]
        reject_sal_prof = reject_sal_prof[~reject_both_prof]
        qc_lev = ds.qc_flags_levels.values
        
        print(' QC: Additional profiles converted to NaNs: ', flush=True)
        print('     >>> {0} temperature profiles '.format(np.sum(reject_tem_prof)), flush=True)
        print('     >>> {0} salinity profiles '.format(np.sum(reject_sal_prof)), flush=True)
        
        reject_tem_lev = np.zeros((ds.dims['profile'], ds.dims['z_dim']), dtype=bool)
        reject_sal_lev = np.zeros((ds.dims['profile'], ds.dims['z_dim']), dtype=bool)
        
        int_tem, int_sal, int_both = self.calculate_all_en4_qc_flags()
        for ii in range(len(int_tem)):
            reject_tem_lev[qc_lev == int_tem[ii]] = 1
        for ii in range(len(int_sal)):
            reject_sal_lev[qc_lev == int_sal[ii]] = 1
        for ii in range(len(int_both)):
            reject_tem_lev[qc_lev == int_both[ii]] = 1
            reject_sal_lev[qc_lev == int_both[ii]] = 1
            
        ds['reject_tem_datapoint'] = (['profile','z_dim'], reject_tem_lev)
        ds['reject_sal_datapoint'] = (['profile','z_dim'], reject_sal_lev)
            
        print('MASKING rejected datapoints, replacing with NaNs:',flush=True)
        ds['temperature'] = xr.where(~reject_tem_lev, ds['temperature'], np.nan)
        ds['potential_temperature'] = xr.where(~reject_tem_lev, ds['temperature'], np.nan)
        ds['practical_salinity'] = xr.where(~reject_tem_lev, ds['practical_salinity'], np.nan)
        
        print('Processed data',flush=True)
        if fn_out is not None:
            print('Writing File: {0}'.format(fn_out), flush=True)
            ds['time'] = (['profile'], pd.to_datetime(ds.time.values))
            ds.set_coords('time')
            general_utils.write_ds_to_file(ds, fn_out, mode='w', unlimited_dims='profile')
            
            print(' >>>>>>>  File Written. ', flush=True)
        
        return_prof = PROFILE()
        return_prof.dataset = ds
        return return_prof
        
    def calculate_all_en4_qc_flags(self):
        '''
        Brute force method for identifying all rejected points according to
        EN4 binary integers. It can be slow to convert large numbers of integers
        to a sequence of bits and is actually quicker to just generate every
        combination of possible QC integers. That's what this routine does.
        Used in PROFILE.preprocess_en4().
        
        INPUTS
         NO INPUTS
         
        OUTPUTS
         qc_integers_tem  : Array of integers signifying the rejection of ONLY
                            temperature datapoints
         qc_integers_sal  : Array of integers signifying the rejection of ONLY
                            salinity datapoints
         qc_integers_both : Array of integers signifying the rejection of BOTH
                            temperature and salinity datapoints.
        '''
        
        reject_tem_ind = 0
        reject_sal_ind = 1
        reject_tem_reasons = [2,3,8,9,10,11,12,13,14,15,16]
        reject_sal_reasons = [2,3,20,21,22,23,24,25,26,27,28,29]
        
        qc_integers_tem = []
        qc_integers_sal = []
        qc_integers_both = []
        n_tem_reasons = len(reject_tem_reasons)
        n_sal_reasons = len(reject_sal_reasons)
        bin_len = 30
        
        # IF reject_tem = 1, reject_sal = 0
        for ii in range(n_tem_reasons):
            bin_tmp = np.zeros(bin_len, dtype=int)
            bin_tmp[reject_tem_ind] = 1
            bin_tmp[reject_tem_reasons[ii]] = 1
            qc_integers_tem.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))
        
        # IF reject_tem = 0, reject_sal = 1
        for ii in range(n_sal_reasons):
            bin_tmp = np.zeros(bin_len, dtype=int)
            bin_tmp[reject_sal_ind] = 1
            bin_tmp[reject_sal_reasons[ii]] = 1
            qc_integers_sal.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))
        
        # IF reject_tem = 1, reject_sal = 1
        for tt in range(n_tem_reasons):
            for ss in range(n_sal_reasons):
                bin_tmp = np.zeros(bin_len, dtype=int)
                bin_tmp[reject_tem_ind] = 1
                bin_tmp[reject_sal_ind] = 1
                bin_tmp[reject_tem_reasons[tt]] = 1
                bin_tmp[reject_sal_reasons[ss]] = 1
                qc_integers_both.append(int("".join(str(jj) for jj in bin_tmp)[::-1], 2))
                
        qc_integers_tem = list( set(qc_integers_tem))  
        qc_integers_sal = list( set(qc_integers_sal)) 
        qc_integers_both = list( set(qc_integers_both)) 
        
        return qc_integers_tem, qc_integers_sal, qc_integers_both
        
        
        
        