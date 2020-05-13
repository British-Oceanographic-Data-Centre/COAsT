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

    def get_subset_of_var(self, var: str, points_x: slice, points_y: slice, line_length: int = None,
                          time_counter: int = 1):
        """
        This method gets a subset of the data across the x/y indices given for the chosen variable.

        :param var: the name of the variable to get data from
        :param points_x: a list/array of indices for the x dimension
        :param points_y: a list/array of indices for the y dimension
        :param line_length: (Optional) the length of your subset (assuming simple line transect)
        :param time_counter: (Optional) which time slice to get data from
        :return: data across all depths for the chosen variable along the given indices
        """
        if line_length is None:
            line_length = len(points_x)

        internal_variable = self.dataset[var].values
        [_, depth_size, _, _, ] = internal_variable.shape

        smaller = np.zeros((depth_size, line_length))

        if time_counter is None:
            for i in range(line_length):
                smaller[:, 1] = internal_variable[:, points_y[i], points_x[i]].squeeze()
        else:
            for i in range(line_length):
                smaller[:, i] = internal_variable[time_counter, :, points_y[i], points_x[i]].squeeze()

        return smaller

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
