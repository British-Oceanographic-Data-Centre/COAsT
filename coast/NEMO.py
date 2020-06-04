from .COAsT import COAsT
import xarray as xa
import numpy as np
from .CDF import CDF
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
        nemo_var = self.dataset[nemo_var_name]
        nemo_time = self.dataset['time_counter']
        
        obs_var = obs_object.dataset[obs_var_name]
        obs_lon = obs_object.longitude
        obs_lat = obs_object.latitude
        obs_time = obs_object.time
    
        # Define output array and check for scalars being used as observations.
        # If so, put obs into lists/arrays.
        n_nh = obs_lon.shape # Number of neighbourhoods (n_nh)  
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
                
            # Create model and observation CDF objects
            model_cdf = CDF(nemo_var_subset, cdf_type=cdf_type)
            obs_cdf = CDF(obs_var[ii], cdf_type=cdf_type)
            
            # Calculate CRPS and put into output array
            crps_list[ii] = self.cdf_diff(model_cdf, obs_cdf)
            
            if plot and n_nh<5:
                plt.figure()
                ax = plt.subplot(111)
                ax.plot(model_cdf.disc_x, model_cdf.disc_y, c='k', 
                        linestyle='--')
                ax.plot(obs_cdf.disc_x, obs_cdf.disc_y, linestyle='--')
                ax.fill_between(model_cdf.disc_x, model_cdf.disc_y, 
                                obs_cdf.disc_y, alpha=0.5)
                plt.title(round( crps_list[ii], 3))
    
        return crps_list
    
    def cdf_diff(self, cdf1: CDF, cdf2: CDF):
        """Calculated the CRPS of provided model and observed CDFs.

        Keyword arguments:
        cdf1 -- Discrete CDF of model data
        cdf2   -- Discrete CDF of observation data
        
        return: A single squared difference between two CDFs.
        """
        xmin = min(cdf1.disc_x[0], cdf2.disc_x[0])
        xmax = max(cdf1.disc_x[-1], cdf2.disc_x[-1])
        common_x = np.linspace(xmin, xmax, 1000)
        cdf1.build_discrete_cdf(x=common_x)
        cdf2.build_discrete_cdf(x=common_x)
        return np.trapz((cdf2.disc_y - cdf1.disc_x)**2, common_x)