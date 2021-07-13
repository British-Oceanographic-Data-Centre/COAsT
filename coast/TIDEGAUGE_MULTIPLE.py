import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import pytz
import sklearn.metrics as metrics
from . import general_utils, plot_util, crps_util, stats_util
from .logging_util import get_slug, debug, error, info
import xarray.ufuncs as uf
import matplotlib.dates as mdates

class TIDEGAUGE_MULTIPLE():
    '''
    This is an object for storage and manipulation of multiple tide gauges
    in a single dataset. This may require some processing of the observations
    such as interpolation to a common time step.
    
    This object's dataset should take the form:
        
        Dimensions:
            port : The locations dimension. Each time series has an index
            time : The time dimension. Each datapoint at each port has an index
            
        Coordinates:
            longitude (port) : Longitude values for each port index
            latitude  (port) : Latitude values for each port index
            time      (port) : Time values for each time index (datetime)
            
    An example data variable could be ssh, or ntr (non-tidal residual). This
    object can also be used for other instrument types, not just tide gauges.
    For example moorings.
            
    Every port index for this object should use the same time coordinates.
    Therefore, timeseries need to be aligned before being placed into the 
    object. If there is any padding needed, then NaNs should be used. NaNs 
    should also be used for quality control/data rejection.
    '''
    def init():
        return
    
    



##############################################################################
###                ~            Plotting             ~                     ###
##############################################################################


##############################################################################
###                ~        Model Comparison         ~                     ###
##############################################################################


##############################################################################
###                ~            Analysis             ~                     ###
##############################################################################
    
    def analyse_ssh(self, fn_out=None, thresholds = np.arange(-.4, 2, 0.1),
                    constit_to_save = ['M2','S2','K2','N2','K1','O1','P1','Q1'], 
                    semidiurnal_constit = ['M2','S2','K2','N2'],
                    diurnal_constit = ['K1','O1','P1','Q1'],
                    apply_ntr_filter = True, dist_omit=100 ):
        '''
        Routine for analysis and comparison of model and observed SSH
        This routine calculates:
            1. Estimates of non-tidal residuals by subtracting an equivalent
               harmonic analysis, i.e. the same time period, missing data,
               constituent sets etc.
            2. Saves some selected harmonic information. Harmonic analysis is done
               using the utide package, with constituent sets based on the
               Rayleigh criterion
            3. NTR stats: MAE, RMSE, correlation, climatology
            4. SSH stats: climatology
            5. Tide stats: RMSE, MAE, correlation, climatology for semidiurnal
               and diurnal frequency bands.
            6. Threshold analysis of NTR. Sum of peaks, time, daily max, monthly
               max over specified thresholds.
        
        INPUTS
         fn_ext               : Filepath to extracted SSH from analyse_ssh
         fn_out               : Filepath to desired output analysis file
         thresholds           : Array of NTR thresholds for analysis
         constit_to_save      : List of constituents to save amplitudes/phase
         semidiurnal_constit  : List of constituents to include in semidiurnal band
         diurnal_constit      : List of constituents to include in diurnal band
         apply_ntr_filter     : If true, apply Savgol filter to non-tidal residuals
                                before analysis.
        '''
        import utide as ut
        import scipy.signal as signal
        
        min_datapoints=744
        
        ds_ssh = self.dataset
        
        # Define Dimension Sizes
        n_port = ds_ssh.dims['port']
        n_time = ds_ssh.dims['time']
        n_constit = len(constit_to_save)
        n_thresholds = len(thresholds)
        seasons = ['DJF','JJA','MAM','SON','All']
        n_seasons = len(seasons)
        freq_bands = ['diurnal', 'semidiurnal', 'all']
        n_freq_bands = len(freq_bands)
        
        # Remove flagged locations
        #ds_ssh.ssh_mod[ds_ssh.bad_flag.values] = np.nan
        #ds_ssh.ssh_obs[ds_ssh.bad_flag.values] = np.nan 
        
        # NTR dataset
        ds_ntr = xr.Dataset(coords = dict(
                                time = ('time', ds_ssh.time.values),
                                longitude = ('port', ds_ssh.longitude.values),
                                latitude = ('port', ds_ssh.latitude.values)),
                            data_vars = dict(
                                ntr_mod = (['port','time'], np.zeros((n_port, n_time))*np.nan),
                                ntr_obs = (['port','time'], np.zeros((n_port, n_time))*np.nan),
                                ntr_err = (['port','time'], np.zeros((n_port, n_time))*np.nan),
                                ntr_square_err = (['port','time'], np.zeros((n_port, n_time))*np.nan),
                                ntr_abs_err = (['port','time'], np.zeros((n_port, n_time))*np.nan)))
        
        ds_tide = xr.Dataset(coords = dict(
                                time = ('time', ds_ssh.time.values),
                                longitude = ('port', ds_ssh.longitude.values),
                                latitude = ('port', ds_ssh.latitude.values),
                                freq_band = ('freq_band', freq_bands)),
                            data_vars = dict(
                                tide_mod = (['port','freq_band','time'], np.zeros((n_port, n_freq_bands, n_time))*np.nan),
                                tide_obs = (['port','freq_band','time'], np.zeros((n_port, n_freq_bands, n_time))*np.nan),
                                tide_err = (['port','freq_band','time'], np.zeros((n_port, n_freq_bands, n_time))*np.nan),
                                tide_square_err = (['port','freq_band','time'], np.zeros((n_port, n_freq_bands, n_time))*np.nan),
                                tide_abs_err = (['port','freq_band','time'], np.zeros((n_port, n_freq_bands,n_time))*np.nan)))
        
        # ANALYSIS dataset
        ds_stats = xr.Dataset(coords = dict(
                        longitude = ('port', ds_ssh.longitude.values),
                        latitude = ('port', ds_ssh.latitude.values),
                        time = ('time', ds_ssh.time.values),
                        season = ('season', seasons),
                        constituent = ('constituent', constit_to_save),
                        threshold = ('threshold', thresholds)),
                   data_vars = dict(
                        a_mod = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        a_obs = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        g_mod = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        g_obs = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        a_err = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        g_err = (['port','constituent'], np.zeros((n_port, n_constit))*np.nan),
                        ssh_std_obs = (['port','season'], np.zeros((n_port, n_seasons))*np.nan),
                        ssh_std_mod = (['port','season'], np.zeros((n_port, n_seasons))*np.nan),
                        ssh_std_err = (['port','season'], np.zeros((n_port, n_seasons))*np.nan),
                        ntr_corr = (['port','season'],  np.zeros((n_port, n_seasons))*np.nan),
                        ntr_mae  = (['port','season'], np.zeros((n_port, n_seasons))*np.nan),
                        ntr_me  = (['port','season'], np.zeros((n_port, n_seasons))*np.nan),
                        ntr_rmse = (['port','season'],  np.zeros((n_port, n_seasons))*np.nan),
                        ntr_err_std = (['port','season'],  np.zeros((n_port, n_seasons))*np.nan),
                        ntr_std_obs = (['port','season'],   np.zeros((n_port, n_seasons))*np.nan),
                        ntr_std_mod = (['port','season'],   np.zeros((n_port, n_seasons))*np.nan),
                        thresh_peak_mod  = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_peak_obs  = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_time_mod = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_time_obs = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_dailymax_mod = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_dailymax_obs = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_monthlymax_mod = (['port', 'threshold'], np.zeros((n_port, n_thresholds))),
                        thresh_monthlymax_obs = (['port', 'threshold'], np.zeros((n_port, n_thresholds))))) 
        
        # Identify seasons
        month_season_dict = {1:0, 2:0, 3:2, 4:2, 5:2, 6:1,
                             7:1, 8:1, 9:3, 10:3, 11:3, 12:0}
        time = ds_ssh.time.values
        pd_time = pd.to_datetime(time)
        pd_month = pd_time.month
        pd_season = [month_season_dict[ii] for ii in pd_month]
        
        # Loop over tide gauge locations, perform analysis per location
        for pp in range(0,n_port):
            
            # Temporary in-loop datasets
            ds_ssh_port = ds_ssh.isel(port=pp).load()
            ssh_mod = ds_ssh_port.ssh_mod
            ssh_obs = ds_ssh_port.ssh_obs
            mask = ds_ssh.mask.isel(port=pp).load().values
            
            if all(uf.isnan(ssh_mod)) or all(uf.isnan(ssh_obs)):
                print('reject 1')
                continue
            
            if sum(~np.isnan(ssh_obs.values))<min_datapoints:
                print('reject 2')
                print(sum(~np.isnan(ssh_obs.values)))
    
            # Harmonic analysis datenums
            ha_time  = mdates.date2num(time)
            
            # Do harmonic analysis using UTide
            uts_obs = ut.solve(ha_time, ssh_obs.values, lat=ssh_obs.latitude.values)
            uts_mod = ut.solve(ha_time, ssh_mod.values, lat=ssh_mod.latitude.values)
            
            # Reconstruct full tidal signal 
            tide_obs = np.array( ut.reconstruct(ha_time, uts_obs).h)
            tide_mod = np.array( ut.reconstruct(ha_time, uts_mod).h)
            tide_obs[mask] = np.nan
            tide_mod[mask] = np.nan
            ds_tide['tide_obs'][pp, -1, :] = tide_obs
            ds_tide['tide_mod'][pp, -1, :] = tide_mod
            
            # Reconstruct partial semidiurnal tidal signal 
            tide_2_obs = np.array( ut.reconstruct(ha_time, uts_obs,
                                                  constit = semidiurnal_constit).h)
            tide_2_mod = np.array( ut.reconstruct(ha_time, uts_mod, 
                                                  constit = semidiurnal_constit).h)
            tide_2_obs[mask] = np.nan
            tide_2_mod[mask] = np.nan
            ds_tide['tide_obs'][pp, 1, :] = tide_2_obs
            ds_tide['tide_mod'][pp, 1, :] = tide_2_mod
            
            # # Reconstruct partial diurnal tidal signal 
            tide_1_obs = np.array( ut.reconstruct(ha_time, uts_obs,
                                                  constit = diurnal_constit).h)
            tide_1_mod = np.array( ut.reconstruct(ha_time, uts_mod,
                                                  constit = diurnal_constit).h)
            tide_1_obs[mask] = np.nan
            tide_1_mod[mask] = np.nan
            ds_tide['tide_obs'][pp, 0, :] = tide_1_obs
            ds_tide['tide_mod'][pp, 0, :] = tide_1_mod
            
            # TWL: SAVE constituents
            a_dict_obs = dict( zip(uts_obs.name, uts_obs.A) )
            a_dict_mod = dict( zip(uts_mod.name, uts_mod.A) )
            g_dict_obs = dict( zip(uts_obs.name, uts_obs.g) )
            g_dict_mod = dict( zip(uts_mod.name, uts_mod.g) )
            
            for cc in range(0, len(constit_to_save)):
                if constit_to_save[cc] in uts_obs.name:
                    ds_stats['a_mod'][pp,cc] = a_dict_mod[constit_to_save[cc]] 
                    ds_stats['a_obs'][pp,cc] = a_dict_obs[constit_to_save[cc]] 
                    ds_stats['g_mod'][pp,cc] = g_dict_mod[constit_to_save[cc]] 
                    ds_stats['g_obs'][pp,cc] = g_dict_obs[constit_to_save[cc]]
            
            # NTR: Calculate non tidal residuals
            ntr_obs = ssh_obs.values - tide_obs
            ntr_mod = ssh_mod.values - tide_mod
            
            # NTR: Apply filter if wanted
            if apply_ntr_filter:
                ntr_obs = signal.savgol_filter(ntr_obs,25,3)
                ntr_mod = signal.savgol_filter(ntr_mod,25,3)
                
            if sum(~np.isnan(ntr_obs)) < min_datapoints:
                continue
                
            ntr_err = ntr_mod - ntr_obs
            ds_ntr['ntr_obs'][pp] = ntr_obs
            ds_ntr['ntr_mod'][pp] = ntr_mod
            ds_ntr['ntr_err'][pp] = ntr_err
            ds_ntr['ntr_abs_err'][pp] = np.abs(ntr_err)
            ds_ntr['ntr_square_err'][pp] = ntr_err**2
            
            # Make masked arrays for seasonal correlation calculation
            ntr_obs = np.ma.masked_invalid(ntr_obs)
            ntr_mod = np.ma.masked_invalid(ntr_mod)
            ds_stats['ntr_corr'][pp,4] = np.ma.corrcoef(ntr_obs, ntr_mod)[1,0]
            for ss in range(0,4):
                season_ind = pd_season == ss
                if np.sum(season_ind)>100:
                    tmp_obs = ntr_obs[season_ind]
                    tmp_mod = ntr_mod[season_ind]
                    ds_stats['ntr_corr'][pp,4] = np.ma.corrcoef(tmp_obs, tmp_mod)[1,0]
                
            # Identify NTR peaks for threshold analysis
            pk_ind_ntr_obs,_ = signal.find_peaks(ntr_obs, distance = 12)
            pk_ind_ntr_mod,_ = signal.find_peaks(ntr_mod, distance = 12)
            pk_ntr_obs = ntr_obs[pk_ind_ntr_obs]
            pk_ntr_mod = ntr_mod[pk_ind_ntr_mod]
            
            # Calculate daily and monthly maxima for threshold analysis
            ds_daily = ds_ntr.groupby('time.day')
            ds_daily_max = ds_daily.max(skipna=True)
            ds_monthly = ds_ntr.groupby('time.month')
            ds_monthly_max = ds_monthly.max(skipna=True)
            
            # Threshold Analysis
            for nn in range(0,n_thresholds):
                threshn = thresholds[nn]
                # NTR: Threshold Frequency (Peaks)
                ds_stats['thresh_peak_mod'][pp, nn] = np.sum( pk_ntr_mod >= threshn)
                ds_stats['thresh_peak_obs'][pp, nn] = np.sum( pk_ntr_obs >= threshn)
                
                # NTR: Threshold integral (Time over threshold)
                ds_stats['thresh_time_mod'][pp, nn] = np.sum( ntr_mod >= threshn)
                ds_stats['thresh_time_obs'][pp, nn] = np.sum( ntr_obs >=threshn)
                
                # NTR: Number of daily maxima over threshold
                ds_stats['thresh_dailymax_mod'][pp, nn] = np.sum( ds_daily_max.ntr_mod.values >= threshn)
                ds_stats['thresh_dailymax_obs'][pp, nn] = np.sum( ds_daily_max.ntr_obs.values >= threshn)
                
                # NTR: Number of monthly maxima over threshold
                ds_stats['thresh_monthlymax_mod'][pp, nn] = np.sum( ds_monthly_max.ntr_mod.values >= threshn)
                ds_stats['thresh_monthlymax_obs'][pp, nn] = np.sum( ds_monthly_max.ntr_mod.values >= threshn)
                
        
        # Seasonal Climatology
        ntr_seasonal = ds_ntr.groupby('time.season')
        ntr_seasonal_std = ntr_seasonal.std(skipna=True)
        ntr_seasonal_mean = ntr_seasonal.mean(skipna=True)
        ssh_seasonal = ds_ssh.groupby('time.season')
        ssh_seasonal_std = ssh_seasonal.std(skipna=True)
        
        sii = 0
        for ss in ntr_seasonal_std['season'].values:
            ind = seasons.index(ss)
            
            ds_stats['ntr_std_mod'][:, ind] = ntr_seasonal_std.ntr_mod.sel(season=ss)
            ds_stats['ntr_std_obs'][:, ind] = ntr_seasonal_std.ntr_obs.sel(season=ss)
            ds_stats['ntr_err_std'][:, ind] = ntr_seasonal_std.ntr_err.sel(season=ss)
            
            ds_stats['ntr_mae'][:, ind] = ntr_seasonal_mean.ntr_abs_err.sel(season=ss)
            ds_stats['ntr_rmse'][:, ind] = np.nanmean( ntr_seasonal_mean.ntr_square_err.sel(season=ss) )
            ds_stats['ntr_me'][:, ind] = ntr_seasonal_mean.ntr_err.sel(season=ss)
            
            ds_stats['ssh_std_mod'][:, ind] = ssh_seasonal_std.ssh_mod.sel(season=ss)
            ds_stats['ssh_std_obs'][:, ind] = ssh_seasonal_std.ssh_obs.sel(season=ss)
            sii+=1
            
        # Annual means and standard deviations
        ntr_std = ds_ntr.std(dim='time', skipna=True)
        ssh_std = ds_ssh.std(dim='time', skipna=True)
        ntr_mean = ds_ntr.mean(dim='time', skipna=True)
        
        ds_stats['ntr_std_mod'][:, 4] = ntr_std.ntr_mod
        ds_stats['ntr_std_obs'][:, 4] = ntr_std.ntr_obs
        ds_stats['ntr_err_std'][:, 4] = ntr_std.ntr_err
        
        ds_stats['ntr_mae'][:, 4] = ntr_mean.ntr_abs_err
        ds_stats['ntr_rmse'][:, 4] = np.nanmean( ntr_mean.ntr_square_err )
        ds_stats['ntr_me'][:, 4] = ntr_mean.ntr_err
        
        ds_stats['ssh_std_mod'][:, 4] = ssh_std.ssh_mod
        ds_stats['ssh_std_obs'][:, 4] = ssh_std.ssh_obs
        
        ds_stats = xr.merge((ds_ssh, ds_ntr, ds_stats, ds_tide))
        
        if fn_out is not None:
            general_utils.write_ds_to_file(ds_stats, fn_out)
            
        tg_out = TIDEGAUGE_MULTIPLE()
        tg_out.dataset = ds_stats
        return tg_out
        
    def extract_ssh(self, nemo, fn_out=None):
                         
        '''
        Routine for extraction of model ssh at obs locations.
        
        The tidegauge file should be a netcdf file with dimension
        ('port', 'time') and variables 'ssh', 'time', 'longitude', 'latitude'.
        
        All ports are expected to have data on a common frequency, so some
        preprocessing of obs is required.
        
        INPUTS
         fn_out          : Absolute path to output file
         
        OUTPUTS
         Writes a new file containing extracted SSH time series, ready for input
         to analyse_ssh(). The output netCDF file contains fimensions:
        '''
        
        nemo = nemo.dataset[['ssh','bottom_level','time_instant']]
        obs = self.dataset
        
        # Get NEMO landmask
        landmask = np.array(nemo.bottom_level.values.squeeze() == 0)
        
        # Subset obs by NEMO domain
        print('Subsetting observation locations to model domain.', flush=True)
        lonmax = np.nanmax(nemo.longitude)
        lonmin = np.nanmin(nemo.longitude)
        latmax = np.nanmax(nemo.latitude)
        latmin = np.nanmin(nemo.latitude)
        ind = general_utils.subset_indices_lonlat_box(obs.longitude, obs.latitude, 
                                           lonmin, lonmax, latmin, latmax)
        obs = obs.isel(port=ind[0])
        print('     >>> Done.', flush=True)
        
        # Determine spatial indices
        print('Calculating spatial indices.', flush=True)
        ind2D = general_utils.nearest_indices_2D(nemo.longitude, nemo.latitude, 
                                                 obs.longitude, obs.latitude,
                                                 mask = landmask)
        print('     >>> Done.', flush=True)
        
        # Extract spatial time series
        print('Calculating time indices.', flush=True)
        nemo_extracted = nemo.isel(x_dim = ind2D[0], y_dim = ind2D[1])
        nemo_extracted = nemo_extracted.swap_dims({'dim_0':'port'})
        print('     >>> Done.', flush=True)
        
        # Compute data (takes a while..)
        print(' Extracting model time series.. ', flush = True)
        nemo_extracted.load()
        print('     >>> Done.', flush=True)
        
        # Check interpolation distances
        print('Calculating interpolation distances.', flush=True)
        n_port = nemo_extracted.dims['port']
        interp_dist = general_utils.calculate_haversine_distance(nemo_extracted.longitude, 
                                                                 nemo_extracted.latitude, 
                                                                 obs.longitude.values,
                                                                 obs.latitude.values)
        print('     >>> Done.', flush=True)
        
        # Align timings
        print('Aligning timings of obs and model', flush=True)
        obs = obs.interp(time = nemo_extracted.time_instant.values, method = 'linear')
        print('     >>> Done.', flush=True)
        
        # Apply shared mask
        print('Applying shared mask.', flush=True)
        ssh_mod = nemo_extracted.ssh.values.T
        ssh_obs = obs.ssh.values
        mask_mod = np.isnan(ssh_mod)
        mask_obs = np.isnan(ssh_obs)
        shared_mask = np.logical_or(mask_mod, mask_obs)
        
        ssh_mod[shared_mask] = np.nan
        ssh_obs[shared_mask] = np.nan
        print('     >>> Done.', flush=True)
        
        print(ssh_mod.shape)
        print(nemo_extracted)
        
        ext = xr.Dataset(coords = dict(
                        longitude = ('port', obs.longitude.values),
                        latitude = ('port', obs.latitude.values),
                        time = ('time', nemo_extracted.time_instant)),
                   data_vars = dict(
                        ssh_mod = (['port','time'], ssh_mod),
                        ssh_obs  = (['port','time'], ssh_obs),
                        mask = (['port','time'], shared_mask),
                        interp_dist = (['port'], interp_dist)))
        
        if fn_out is not None:
            print('Writing to file..', flush=True)
            general_utils.write_ds_to_file(ext, fn_out)
            print('     >>> Done.', flush=True)
            
        tg_out = TIDEGAUGE_MULTIPLE()
        tg_out.dataset = ext
        return tg_out