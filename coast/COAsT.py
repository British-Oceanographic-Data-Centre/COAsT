from dask import delayed
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
import datetime

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

    def normal_distribution(self, x=np.arange(-6,6,0.001), mu=0, sigma=1):
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

    def cumulative_distribution(self, x, pdf):
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

    def crps(self, x, model_cdf, obs_cdf):
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

    def empirical_distribution(self, x, sample):
        """Estimates a CDF empirically.

        Keyword arguments:
        x      -- Array of x-values over which to generate distribution
        sample -- Sample to use to generate distribution
        
        return: Array of len(x) containing corresponding EDF values
        """
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        edf = np.zeros(len(x))
        n_sample = len(sample)
        for ss in sample:
            edf[x>ss] = edf[x>ss] + 1/n_sample
        return xr.DataArray(edf)

    def extract_lonlat_box(self, lon, lat, lonbounds, latbounds):
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
    
    def num_to_date(self, dnum, epoch = datetime.datetime(1900,1,1), 
                    units = 'days'):
        """ Converts a datenumber to a datetime object. 
        
        Default is dnum is 'days since 1900-01-01' but the units and epoch can 
        be specified if needed. Works for either arraylike objects or scalars.
        
        Keyword arguments:
        dnum  -- datenumber or array of datenumbers to convert to datetime. 
        epoch -- Datetime with which the provided datenumber is relative to.
        units -- Units of datenumber.
        
        return: datetime or array of datetime objects
        """
    
        adj_dict = {"days" : 1, "hours" : 24, 
                    "minutes" : 24*60, "seconds": 24*60*60}
    
        if np.isscalar(dnum):
            dnum = dnum/adj_dict[units]
            dtime = epoch + datetime.timedelta(days=dnum) 
        else:
            dnum = dnum/adj_dict[units]
            dtime = np.array( [epoch + datetime.timedelta(days=dii) \
                               for dii in dnum],
                             dtype = np.datetime64 )
            
        return dtime

    def date_to_num(self, dtime, epoch = datetime.datetime(1900,1,1),
                    units = 'days'):
        """ Converts a datetime object to a datenumber.
        
        Default dnum is 'days since 1900-01-01' but the units and epoch can 
        be specified if needed. Works for either arraylike objects or scalars.
        
        Keyword arguments:
        dtime -- Datetime object or array of datetime objects to convert.
        epoch -- Datetime with which the provided datenumber is relative to.
        units -- Units of datenumber.
        
        return: Number of specified units since specified epoch.
        """
        adj_dict = {"days" : 24*60*60, "hours" : 60*60, 
                    "minutes" : 60, "seconds": 1}
    
        if isinstance(dtime, (list, np.ndarray, tuple)):
            datenum = np.array( [(dii - epoch).total_seconds() \
                                 for dii in dtime])
        else:
            datenum = (dtime - epoch).total_seconds()
    
        return datenum/(adj_dict[units])
