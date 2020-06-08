from .COAsT import COAsT
import xarray as xa
import numpy as np
# from dask import delayed, compute, visualize
# import graphviz
import matplotlib.pyplot as plt

class NEMO(COAsT):

    def __init__(self):
        super()
        self.ssh = None
        self.nav_lat = None
        self.nav_lon = None
        self.botpres = None
        self.toce = None
        self.soce = None
        self.e3t = None
        self.e3u = None
        self.e3v = None
        self.uoce = None
        self.voce = None
        self.utau = None
        self.vtau = None

    def set_command_variables(self):
        """ A method to make accessing the following simpler
                ssh (t,y,x) - sea surface height above geoid - (m)
                botpres (t,y,x) - sea water pressure at sea ﬂoor - (dbar)
                toce (t,z,y,x) -  sea water potential temperature -  (degC)
                soce (t,z,y,x) - sea water practical salinity - (degC)
                e3t (t,z,y,x) - T-cell thickness - (m)
                e3u (t,z,y,x) - U-cell thickness - (m)
                e3v (t,z,y,x) - V-cell thickness - (m)
                uoce (t,z,y,x) - sea water x-velocity (m/s)
                voce (t,z,y,x) - sea water y-velocity (m/s)
                utau(t,y,x) - wind stress x (N/m2)
                vtau(t,y,x) - wind stress y (N/m2)
        """
        try:
            self.nav_lon = self.dataset.nav_lon
        except AttributeError as e:
            print(str(e))

        try:
            self.nav_lat = self.dataset.nav_lat
        except AttributeError as e:
            print(str(e))

        try:
            self.ssh = self.dataset.sossheig
        except AttributeError as e:
            print(str(e))

        try:
            self.botpres = self.dataset.botpres
        except AttributeError as e:
            print(str(e))

        try:
            self.toce = self.dataset.voctemper
        except AttributeError as e:
            print(str(e))

        try:
            self.soce = self.dataset.soce
        except AttributeError as e:
            print(str(e))

        try:
            self.e3t = self.dataset.e3t
        except AttributeError as e:
            print(str(e))
        try:
            self.e3u = self.dataset.e3u
        except AttributeError as e:
            print(str(e))
        try:
            self.e3v = self.dataset.e3v
        except AttributeError as e:
            print(str(e))
        try:
            self.uoce = self.dataset.uoce
        except AttributeError as e:
            print(str(e))
        try:
            self.voce = self.dataset.voce
        except AttributeError as e:
            print(str(e))
        try:
            self.utau = self.dataset.utau
        except AttributeError as e:
            print(str(e))
        try:
            self.vtau = self.dataset.vtau
        except AttributeError as e:
            print(str(e))

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller

    def crps_sonf(self, nemo_var_name, nemo_dom, obs_object, obs_var_name,
                  nh_radius=111, nh_type = "radius", cdf_type = "empirical",
                  time_interp = "nearest", plot=False):
        """Calculatues the Continuous Ranked Probability Score (CRPS)
    
        Calculatues the Continuous Ranked Probability Score (CRPS) using
        a single-observation and neighbourhood forecast (SONF). The statistic
        uses a comparison between the probability distributions of a model 
        neighbourhood subset and a single observation. The CRPS is calculated 
        independently for each observation. 

        Keyword arguments:
        nemo_var_name -- COAsT variable string.
        nemo_dom -- COAsT DOMAIN object
        obs_lon -- Array of observation longitudes
        obs_lat -- Array of observation latitudes
        obs_var -- Array of observation variables
        nh_radius -- Neighbourhood radius in km (if radius method) or degrees 
                     (if box method).
        nh_type -- Neighbourhood determination method: 'radius' or 'box'.
        cdf_type -- Method for model CDF determination: 'empirical' or 
                    'theoretical'. Observation CDFs are always determined 
                    empirically.
        time_interp -- Method for interpolating in time, currently only
                       "none" and "nearest". For none, only a single model
                       time slice should be supplied and observations are
                       assumed to correspond with the slice correctly.
        plot -- True or False. Will plot up to five CDF comparisons and CRPS.
        
        return: Array of CRPS scores for each observation supplied.
        """
        # Define var_dict to determine which variable to use and define some
        # function variables
        nemo_var = getattr(self, nemo_var_name)
        nemo_time = self.dataset.time_counter
        
        obs_var = getattr(obs_object, obs_var_name)
        obs_lon = obs_object.longitude
        obs_lat = obs_object.latitude
        obs_time = obs_object.time
    
        # Define output array and check for scalars being used as observations.
        # If so, put obs into lists/arrays.
        n_nh = obs_var.shape # Number of neighbourhoods (n_nh)  
        if len(n_nh) == 0: #Scalar case
            n_nh = 1
            obs_lon  =  [obs_lon]
            obs_lat  =  [obs_lat]
            obs_var  =  [obs_var]
            obs_time =  [obs_time]
        else:
            n_nh = n_nh[0]
        crps_list = np.zeros( n_nh )
        # Loop over neighbourhoods
        for ii in range(0, n_nh):
        
            # Neighbourhood centre
            cntr_lon = obs_lon[ii]
            cntr_lat = obs_lat[ii]
        
            # Get model neighbourhood subset using specified method
            if nh_type == "radius":
                subset_indices = nemo_dom.subset_indices_by_distance(cntr_lon, 
                                 cntr_lat, nh_radius)
            elif nh_type == "box":
                lonbounds = [ cntr_lon - nh_radius, cntr_lon + nh_radius ]
                latbounds = [ cntr_lat - nh_radius, cntr_lat + nh_radius ]
                subset_indices = nemo_dom.subset_indices_lonlat_box(lonbounds, 
                                                                    latbounds )
            # Subset model data in time and space
            if time_interp == "nearest": 
                # CURRENTLY DOES NOTHING, TAKES FIRST INDEX
                time_ind = 0
                nemo_var_subset = nemo_var[time_ind,
                                           xa.DataArray(subset_indices[0]), 
                                           xa.DataArray(subset_indices[1])]
        
            if nemo_var_subset.shape[0] == 0:
                raise ValueError('Model neighbourhood contains no points.' + 
                                 ' Try increasing neighbourhood size.')
            # Calculate model cumulative distribution function
            model_mu = np.nanmean(nemo_var_subset)
            model_sigma = np.nanmean(nemo_var_subset)
            cdf_x = np.arange( model_mu - 5 * model_sigma, 
                                model_mu + 5 * model_sigma, model_sigma / 100 )
            if cdf_type == "empirical":
                model_cdf = self.empirical_distribution(cdf_x, 
                                                   nemo_var_subset)
            elif cdf_type == "theoretical": # TODO: Add more distributions
                model_pdf = self.normal_distribution(cdf_x, mu = model_mu, 
                                                sigma=model_sigma)
                model_cdf = self.cumulative_distribution(cdf_x, model_pdf)
            
            # Calculate observation empirical distribution function
            obs_cdf = self.empirical_distribution(cdf_x, obs_var[ii])
            
            # Calculate CRPS and put into output array
            crps_list[ii] = self.crps(cdf_x, model_cdf, obs_cdf)
            if plot and n_nh<5:
                plt.figure()
                plt.plot(cdf_x, model_cdf, c='k', linestyle='--')
                plt.plot(cdf_x, obs_cdf, linestyle='--')
                plt.fill_between(cdf_x, model_cdf, obs_cdf, alpha=0.5)
                plt.title(round( crps_list[ii], 3))
    
        return crps_list
    
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
