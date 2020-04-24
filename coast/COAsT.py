from dask import delayed
from dask import array
import xarray as xr
from dask.distributed import Client


class COAsT:
    def __init__(self, workers=2, threads=2, memory_limit_per_worker='2GB'):
        # self.client = Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)
        self.dataset = None

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
