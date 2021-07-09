import numpy as np
import xarray as xr
from . import general_utils, plot_util, crps_util, COAsT
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
        > level   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files. 
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

    def extract_daily_ts_per_file(self, nemo, fn_out, run_name = 'Undefined', 
                            z_interp = 'linear',
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
        
        This will apply EN4 QC Flags to the observation data as follows:
            1. Completely rejected profiles (both temp and sal) are removed
               entirely and will not appear in the output dataset.
            2. Where a complete tem/sal profile is rejected but the other isnt,
               all depths are set to nan for that variable.
            3. For remaining profiles, the level-by-level QC flags are applied.
               Where data at a level is rejected, it is set to NaN.
        The output dataset will not contain rejected values, only NaNs.
        
        INPUTS:
         nemo                 : NEMO object created on t-grid
         fn_out (str)         : Absolute filepath for desired output file.
         run_name (str)       : Name of run. [default='Undefined']
                                Will be saved to output
         z_interp (str)       : Type of scipy interpolation to use for depth interpolation
                                [default = 'linear']
         en4_sal_name (str)   : Name of EN4 (COAST) variable to use for salinity
                                [default = 'practical_salinity']
         en4_tem_name (str)   : Name of EN4 (COAST) variable to use for temperature
                                [default = 'potential_temperature']
         fn_mesh_mask (str)   : Full path to mesh_mask.nc if data is NEMO v3.6
                                
        OUTPUTS:
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
                           'qc_flags_profiles', 'qc_flags_levels']]
        en4 = en4.rename({en4_sal_name:'sal', en4_tem_name:'tem'})
        
        # 3) Use only observations that are within model domain
        lonmax = np.nanmax(nemo['longitude'])
        lonmin = np.nanmin(nemo['longitude'])
        latmax = np.nanmax(nemo['latitude'])
        latmin = np.nanmin(nemo['latitude'])
        ind = general_utils.subset_indices_lonlat_box(en4['longitude'], 
                                                            en4['latitude'],
                                                            lonmin, lonmax, 
                                                            latmin, latmax)[0]
        
        print(' ', flush=True)
        en4 = en4.isel(profile=ind)
        print('EN4 subsetted to model domain: ', flush=True)
        print('    >>> LON {0} -> {1}'.format(str(lonmin), str(lonmax)), flush=True)
        print('    >>> LAT {0} -> {1}'.format(str(latmin), str(latmax)), flush=True)
        
        # 4) Use only observations that are within model time window.
        en4_time = en4.time.values
        time_max = pd.to_datetime( max(mod_time) ) + relativedelta(hours=12)
        time_min = pd.to_datetime( min(mod_time) ) - relativedelta(hours=12)
        ind = np.logical_and( en4_time >= time_min, en4_time <= time_max )
        en4 = en4.isel(profile=ind)
        print('EN4 subsetted to model time period:', flush=True)
        print('    >>> {0} -> {1}'.format(str(time_min), str(time_max)), flush=True)
        print('  ', flush=True)
        
        # REJECT profiles that are QC flagged.
        print(' Applying QUALITY CONTROL to EN4 data...', flush=True)
        en4.qc_flags_profiles.load()
        
        # This line reads converts the QC integer to a binary string.
        # Each bit of this string is a different QC flag. Which flag is which can
        # be found on the EN4 website:
        # https://www.metoffice.gov.uk/hadobs/en4/en4-0-2-profile-file-format.html
        qc_str = [np.binary_repr(en4.qc_flags_profiles.values[pp]).zfill(30)[::-1] for pp in range(en4.dims['profile'])]
        
        # Determine indices of kept profiles
        reject_tem = np.array( [int( qq[0] ) for qq in qc_str], dtype=bool )
        reject_sal = np.array( [int( qq[1] ) for qq in qc_str], dtype=bool )
        reject_both = np.logical_and( reject_tem, reject_sal )
        keep_both = ~reject_both
        print('     >>> QC: Completely rejecting {0} / {1} profiles'.format(np.sum(reject_both), en4.dims['profile']), flush=True)
        
        en4 = en4.isel(profile=keep_both)
        reject_tem = reject_tem[keep_both]
        reject_sal = reject_sal[keep_both]
        
        print(' QC: Additional profiles converted to NaNs: ', flush=True)
        print('     >>> {0} temperature profiles '.format(np.sum(reject_tem)), flush=True)
        print('     >>> {0} salinity profiles '.format(np.sum(reject_sal)), flush=True)
        
        # LOAD all remaining EN4 data
        en4.load()
    
        # Get model indices (space and time) corresponding to observations
        # Does a basic nearest neighbour analysis in time and space.
        
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
        print('Indexing model data. May take a while..')
        mod_profiles.load()
        print('Model indexed and loaded', flush=True)
        
        # Define variable arrays for interpolated data for monthly EN4 data
        n_mod_levels = mod_profiles.dims['z_dim']
        n_obs_levels = en4.dims['z_dim']
        n_prof = en4.dims['profile']
        data = xr.Dataset(coords = dict(
                              longitude=     (["profile"], en4.longitude.values),
                              latitude=      (["profile"], en4.latitude.values),
                              time=          (["profile"], en4.time.values),
                              level=         (['level'], np.arange(0,n_obs_levels)),
                              ex_longitude = (["profile"], mod_profiles.longitude.values),
                              ex_latitude =  (["profile"], mod_profiles.latitude.values),
                              ex_time =      (["profile"], mod_profiles.time.values),
                              ex_level =     (["ex_level"], np.arange(0, n_mod_levels))),
                          data_vars = dict(
                              mod_tem = (['profile','level'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_tem = (['profile','level'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              mod_sal = (['profile','level'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_sal = (['profile','level'],  np.zeros((n_prof , n_obs_levels))*np.nan),
                              obs_z =    (['profile','level'],    np.zeros((n_prof , n_obs_levels))*np.nan),
                              ex_mod_tem = (["profile", "ex_level"], mod_profiles.tem.values),
                              ex_mod_sal = (["profile", "ex_level"], mod_profiles.sal.values),
                              ex_depth =   (["profile", "ex_level"], mod_profiles.depth_0.values.T),
                              nn_ind_x = (["profile"], ind2D[0]),
                              nn_ind_y = (["profile"], ind2D[1]),
                              bathy = (['profile'], mod_profiles.bathymetry.values),
                              original_dim_x = nemo.dims['x_dim'],
                              original_dim_y = nemo.dims['y_dim']))
        
        
        data['obs_tem'][~reject_tem] = en4.tem.isel(profile=~reject_tem).values
        data['obs_sal'][~reject_sal] = en4.tem.isel(profile=~reject_sal).values
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
            if ~reject_tem[pp]:
                
                # Interpolate model to obs levels
                f = interp.interp1d(mod_profile.depth_0.values, 
                                    mod_profile.tem.values, fill_value=np.nan, 
                                    bounds_error=False, kind = z_interp)
                data['mod_tem'][pp, :obs_bl[pp]] = f( obs_profile.depth.values )
                
            if ~reject_sal[pp]:
                
                # Interpolate model to obs levels
                f = interp.interp1d(mod_profile.depth_0.values, 
                                    mod_profile.sal.values, fill_value=np.nan, 
                                    bounds_error=False, kind = z_interp)
                data['mod_sal'][pp, :obs_bl[pp]] = f( obs_profile.depth.values )
                
            # Figure out levels where no QC is needed
            no_qc_needed = obs_profile.qc_flags_levels.values == 0
                
            # Process level-by-level QC flags
            # IF zero, no problems with level. IF not zeros, determine problem
            # (TEM or SAL or BOTH rejected?) and set NaN
            if np.sum(~no_qc_needed) > 0:
                obs_qc = obs_profile.isel(z_dim = ~no_qc_needed)
                z_indices = z_indices[~no_qc_needed]
                obs_qc_flags = obs_qc.qc_flags_levels.values
                qc_tmp = [np.binary_repr(qq).zfill(30)[::-1] for qq in obs_qc_flags]
                
                reject_tem_tmp = np.array( [int( qq[0] ) for qq in qc_tmp], dtype=bool )
                reject_sal_tmp = np.array( [int( qq[1] ) for qq in qc_tmp], dtype=bool )
                
                if np.sum(reject_tem_tmp) > 0 and ~reject_tem[pp]:
                    z_indices_tmp = z_indices[reject_tem_tmp]
                    data['obs_tem'][pp][z_indices_tmp] = np.nan
                    data['mod_tem'][pp][z_indices_tmp] = np.nan
                    
                if np.sum(reject_sal_tmp) > 0 and ~reject_sal[pp]:
                    z_indices_tmp = z_indices[reject_sal_tmp]
                    data['obs_sal'][pp][z_indices_tmp] = np.nan
                    data['mod_sal'][pp][z_indices_tmp] = np.nan
            
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
        data["error_tem"] = (['profile','level'], data.mod_tem - data.obs_tem)
        data["error_sal"] = (['profile','level'], data.mod_sal - data.obs_sal)
        
        # Absolute errors at all depths
        data["abs_error_tem"] = (['profile','level'], np.abs(data.error_tem))
        data["abs_error_sal"] = (['profile','level'], np.abs(data.error_sal))
        
        # Mean errors across depths
        data['me_tem'] = ('profile', np.nanmean(data.error_tem, axis=1))
        data['me_sal'] = ('profile', np.nanmean(data.error_sal, axis=1))
        
        # Mean absolute errors across depths
        data['mae_tem'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
        data['mae_sal'] = (['profile'], np.nanmean(data.abs_error_tem, axis=1))
        
        # Save REJECT vectors
        data['reject_tem'] = (['profile'], reject_tem)
        data['reject_sal'] = (['profile'], reject_sal)
                  
        # Write monthly stats to file
        general_utils.write_ds_to_file(data, fn_out, mode='w', unlimited_dims='profile')
        
        print(' >>>>>>>  File Written: ' + fn_out, flush=True)
        
    def analyse_daily_ts_regional(self, ds_ext, fn_out, ref_depth,
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
                     'interp_lag', 'reject_tem', 'reject_sal', 'bathy']]
        print('Loading required data..')
        ds.load()
        
        # Restrict time if required or define start and end dates
        print('Constraining dates to {0} -> {1}'.format(start_date, end_date))
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
                
                if ~prof.reject_sal:
                    mask = np.isnan( prof.obs_sal )
                    prof['mod_sal'][mask] = np.nan
                    f = interp.interp1d(prof.obs_z.values, prof.mod_sal.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['mod_sal'][pp] = f( ref_depth )
                    f = interp.interp1d(prof.obs_z.values, prof.obs_sal.values, 
                                        fill_value=np.nan, bounds_error=False)
                    ds_interp['obs_sal'][pp] = f( ref_depth )
                
                if ~prof.reject_tem:
                    mask = np.isnan( prof.obs_tem )
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
        print('Writing to file..')
        general_utils.write_ds_to_file(ds_interp, fn_out)
        print(' >>>>>>>  File Written: ' + fn_out, flush=True)