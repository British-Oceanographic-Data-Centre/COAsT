from dask import delayed
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client


def setup_dask_clinet(workers=2, threads=2, memory_limit_per_worker='2GB'):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class COAsT:
    def __init__(self, workers=2, threads=2, memory_limit_per_worker='2GB'):
        #self.client = Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)
        self.dataset = None
        # Radius of the earth in km
        self.earth_raids = 6371.007176

    def load(self, file, chunks: dict = None):
        self.dataset = xr.open_dataset(file, chunks=chunks)

    def load_multiple(self, directory_to_files, chunks: dict = None):
        self.dataset = xr.open_mfdataset(
            directory_to_files, chunks=chunks, parallel=True, combine="by_coords", compat='override'
        )

    def subset(self, domain, nemo, points_a: array, points_b: array):
        raise NotImplementedError

    def distance_between_two_points(self):
        raise NotImplementedError

    def dist_haversine(self, lon1, lat1, lon2, lat2):
        '''
        # Estimation of geographical distance using the Haversine function.
        # Input can be single values or 1D arrays of locations. This
        # does NOT create a distance matrix but outputs another 1D array.
        # This works for either location vectors of equal length OR a single loc
        # and an arbitrary length location vector.
        #
        # lon1, lat1 :: Location(s) 1.
        # lon2, lat2 :: Location(s) 2.
        '''

        # Convert to radians for calculations
        lon1 = xr.ufuncs.deg2rad(lon1)
        lat1 = xr.ufuncs.deg2rad(lat1)
        lon2 = xr.ufuncs.deg2rad(lon2)
        lat2 = xr.ufuncs.deg2rad(lat2)

        # Latitude and longitude differences
        dlat = (lat2 - lat1) / 2
        dlon = (lon2 - lon1) / 2

        # Haversine function.
        distance = xr.ufuncs.sin(dlat) ** 2 + xr.ufuncs.cos(lat1) * xr.ufuncs.cos(lat2) * xr.ufuncs.sin(dlon) ** 2
        distance = 2 * 6371.007176 * xr.ufuncs.arcsin(xr.ufuncs.sqrt(distance))

        return distance

    def plot_single(self, variable: str):
        return self.dataset[variable].plot()
        # raise NotImplementedError

    def plot_cartopy(self):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        ax = plt.axes(projection=ccrs.Orthographic(5, 15))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        tmp = self.dataset.votemper
        tmp.attrs = self.dataset.votemper.attrs
        tmp.isel(time_counter=0, deptht=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        plt.show()

    def plot_movie(self):
        raise NotImplementedError
        
    def crps_sonf(self, nemo_dom, nemo_var, obs_lon, obs_lat, obs_var,
              nh_radius=111, nh_method = 'radius', cdf_type = 'empirical',
              plot=False):
        """Calculatues the Continuous Ranked Probability Score (CRPS)
    
        Calculatues the Continuous Ranked Probability Score (CRPS) using
        a single-observation and neighbourhood forecast (SONF). The statistic
        uses a comparison between the probability distributions of a model 
        neighbourhood subset and a single observation. The CRPS is calculated 
        independently for each observation. 

        Keyword arguments:
        nemo_dom -- COAsT DOMAIN object
        nemo_var -- COAsT variable object (xa.DataArray) at pre specified time.
        obs_lon -- Array of observation longitudes
        obs_lat -- Array of observation latitudes
        obs_var -- Array of observation variables
        nh_radius -- Neighbourhood radius in km (if radius method) or degrees 
                     (if box method).
        nh_method -- Neighbourhood determination method: 'radius' or 'box'.
        cdf_type -- Method for model CDF determination: 'empirical' or 
                    'theoretical'. Observation CDFs are always determined 
                    empirically.
        plot -- True or False. Will plot up to five CDF comparisons and CRPS.
        
        return: Array of CRPS scores for each observation supplied.
        """
        
        # Cast obs to array if single value is given
        if np.isscalar(obs_lon):
            obs_lon = np.array([obs_lon])
            obs_lat = np.array([obs_lat])
            obs_var = np.array([obs_var])
    
        # Define output array
        n_nh = len(obs_var) # Number of neighbourhoods (nh)
        crps_list = np.zeros( n_nh )
    
        # Loop over variables
        for ii in range(0, n_nh):
        
            cntr_lon = obs_lon[ii]
            cntr_lat = obs_lat[ii]
        
            # Get model subset using specified method
            if nh_method == 'radius':
                subset_indices = nemo_dom.subset_indices_by_distance(cntr_lon, 
                                 cntr_lat, nh_radius)
            elif nh_method == 'box':
                lonbounds = [ cntr_lon - nh_radius, cntr_lon + nh_radius ]
                latbounds = [ cntr_lat - nh_radius, cntr_lat + nh_radius ]
                subset_indices = self.extract_lonlat_box(nemo_dom.dataset.nav_lon,
                                                    nemo_dom.dataset.nav_lat,
                                                    lonbounds, latbounds )
            nemo_var_subset = nemo_var[xr.DataArray(subset_indices[0]), 
                                       xr.DataArray(subset_indices[1])]
        
            # Calculate model cumulative distribution function
            model_mu = np.nanmean(nemo_var_subset)
            model_sigma = np.nanmean(nemo_var_subset)
            cdf_x = np.arange( model_mu - 5 * model_sigma, 
                                model_mu + 5 * model_sigma, model_sigma / 100 )
            if cdf_type == 'empirical':
                model_cdf = self.empirical_distribution(cdf_x, 
                                                   np.array(nemo_var_subset))
            elif cdf_type == 'theoretical':
                model_pdf = self.normal_distribution(cdf_x, mu = model_mu, 
                                                sigma=model_sigma)
                model_cdf = self.cumulative_distribution(cdf_x, model_pdf)
            
            # Calculate observation empirical distribution function
            obs_cdf = self.empirical_distribution(cdf_x, obs_var[ii])
            
            # Calculate CRPS and put into output array
            crps_list[ii] = self.crps(cdf_x, model_cdf, obs_cdf)
            #if plot and n_nh<5:
            #    plt.figure()
            #    plt.plot(cdf_x, model_cdf, c='k', linestyle='--')
            #    plt.plot(cdf_x, obs_cdf, linestyle='--')
            #    plt.fill_between(cdf_x, model_cdf, obs_cdf, alpha=0.5)
            #    plt.title(round( crps_list[ii], 3))
    
        return crps_list

    def normal_distribution(x=np.arange(-6,6,0.001), mu=0, sigma=1):
        """Generates a normal distribution.

        Keyword arguments:
        x     -- Arbitrary array of x-values
        mu    -- Distribution mean
        sigma -- Distribution standard deviation
        
        return: Array of len(x) containing the normal values calculated from
                the elements of x.
        """
        term1 = sigma*np.sqrt( 2*np.pi )
        term1 = 1/term1
        exponent = -0.5*((x-mu)/sigma)**2
        return term1*np.exp( exponent )

    def cumulative_distribution(x, pdf):
        """Estimates the cumulative distribution of a supplied PDF.

        Keyword arguments:
        x   -- Arbitrary array of x-values
        pdf -- PDF corresponding to values in x. E.g. as generated using
               normal_distribution.
        
        return: Array of len(x) containing the discrete cumulative values 
                estimated using the integral under the provided PDF.
        """
        cdf = [np.trapz(pdf[:ii],x[:ii]) for ii in range(0,len(x))]
        return np.array(cdf)

    def crps(x, model_cdf, obs_cdf):
        """Calculated the CRPS of provided model and observed CDFs.

        Keyword arguments:
        x         -- Arbitrary array of x-values. Model_cdf and obs_cdf should
                     share the same x array,
        model_cdf -- Discrete CDF of model data
        obs_cdf   -- Discrete CDF of observation data
        
        return: A single CRPS value.
        """
        diff = model_cdf - obs_cdf
        diff = diff**2
        crps = np.trapz(diff, x)
        return crps

    def empirical_distribution(x, sample):
        """Estimates a CDF empirically.

        Keyword arguments:
        x      -- Array of x-values over which to generate distribution
        sample -- Sample to use to generate distribution
        
        return: Array of len(x) containing corresponding EDF values
        """
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        edf = np.zeros(len(x))
        n_sample = len(sample)
        for ss in sample:
            edf[x>ss] = edf[x>ss] + 1/n_sample
        return edf

    def extract_lonlat_box(lon, lat, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.

        Keyword arguments:
        lon       -- Longitudes, 1D or 2D
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude, max_longitude]
        latbounds -- Array of form [min_latitude, max_latitude]
        
        return: Indices corresponding to datapoints inside specified box
        """
        ff1 = ( lon > lonbounds[0] ).astype(int)
        ff2 = ( lon < lonbounds[1] ).astype(int)
        ff3 = ( lat > latbounds[0] ).astype(int)
        ff4 = ( lat < latbounds[1] ).astype(int)
        ff = ff1 * ff2 * ff3 * ff4
        return np.where(ff)
