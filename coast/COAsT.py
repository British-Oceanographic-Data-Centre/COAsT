from dask import delayed
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
from warnings import warn


def setup_dask_clinet(workers=2, threads=2, memory_limit_per_worker='2GB'):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class COAsT:
    def __init__(self, workers=2, threads=2, memory_limit_per_worker='2GB'):
        # self.client = Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)
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

    def get_subset_as_xarray(self, var: str, points_x: slice, points_y: slice, line_length: int = None,
                             time_counter: int = 1):
        """

        :param var:
        :param points_x:
        :param points_y:
        :param line_length:
        :param time_counter:
        :return:
        """

        try:
            [time_size, depth_size, _, _] = self.dataset[var].shape
            if time_size == 1:
                time_counter == 0

        except ValueError:
            time_counter = None
            [depth_size, _, _] = self.dataset[var].shape

        dx = xr.DataArray(points_x)
        dy = xr.DataArray(points_y)

        if time_counter is None:
            smaller = self.dataset[var].isel(x=dx, y=dy)
        else:
            smaller = self.dataset[var].isel(time_counter=0, x=dx, y=dy)

        return smaller

    def get_subset_of_var(self, var: str, points_x: slice, points_y: slice, line_length: int = None,
                          time_counter: int = 1):
        """
        This method gets a subset of the data across the x/y indices given for the chosen variable.

        Setting time_counter to None will treat `var` as only having 3 dimensions depth, y, x

        there is a check on `var` to see the size of the time_counter, if 1 then time_counter is fixed to index 0.

        :param var: the name of the variable to get data from
        :param points_x: a list/array of indices for the x dimension
        :param points_y: a list/array of indices for the y dimension
        :param line_length: (Optional) the length of your subset (assuming simple line transect)
        :param time_counter: (Optional) which time slice to get data from, if None and the variable only has one a time
                             channel of length 1 then time_counter is fixed too an index of 0
        :return: data across all depths for the chosen variable along the given indices
        """
        # TODO do we need a 3d version of this method - i.e no depth/time channel
        if line_length is None:
            line_length = len(points_x)

        internal_variable = self.dataset[var].values

        # This will fail for a 3d variable, so we retry without time_size
        try:
            [time_size, depth_size, _, _] = internal_variable.shape
            if time_size == 1:
                time_counter == 0

        except ValueError:
            time_counter = None
            [depth_size, _, _] = internal_variable.shape

        smaller = np.zeros((depth_size, line_length))

        if time_counter is None:
            for i in range(line_length):
                smaller[:, 1] = internal_variable[:, points_y[i], points_x[i]].squeeze()
        else:
            for i in range(line_length):
                smaller[:, i] = internal_variable[time_counter, :, points_y[i], points_x[i]].squeeze()

        return smaller

    def plot_simple_2d(self, x, y, data: xr.DataArray, cmap):
        import matplotlib.pyplot as plt

        plt.close('all')

        fig = plt.figure()
        plt.rcParams['figure.figsize'] = (15.0, 15.0)

        ax = fig.add_subplot(411)
        plt.pcolormesh(x, y, data, cmap=cmap)

        plt.ylim([200, 0])
        plt.xlim([-10, 0])
        plt.title('Temperature')
        plt.ylabel('depth (m)')
        plt.clim([8, 15])
        plt.colorbar()

        return plt

    def plot_cartopy(self, var: str, plot_var: array, params, time_counter: int = 0):
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
