"""Timeseries Class"""
from .index import Indexed
from . import general_utils


class Timeseries(Indexed):
    """Parent class for Tidegauge and other timeseries type datasets
    Common methods ...
    """

    pass

    def obs_operator(self, gridded, time_interp="nearest"):
        """ 
        Maps a gridded object time series locations.
        
        
        """
        gridded = gridded.dataset
        ds = self.dataset

        # Determine spatial indices
        print("Calculating spatial indices.", flush=True)
        
        # Determine if landmask is available
        if 'landmask' in list(gridded.keys()):
            landmask = gridded.landmask
        else:
            landmask = None
            
            
        ind_x, ind_y = general_utils.nearest_indices_2d(
            gridded.longitude, gridded.latitude, ds.longitude, ds.latitude, 
            mask=landmask
        )

        # Extract spatial time series
        print("Calculating time indices.", flush=True)
        extracted = gridded.isel(x_dim=ind_x, y_dim=ind_y)
        extracted = extracted.swap_dims({"dim_0": "id"})

        # Compute data (takes a while..)
        print(" Indexing model data at tide gauge locations.. ", flush=True)
        extracted.load()

        # Check interpolation distances
        print("Calculating interpolation distances.", flush=True)
        interp_dist = general_utils.calculate_haversine_distance(
            extracted.longitude, extracted.latitude, ds.longitude.values, ds.latitude.values
        )

        # Interpolate model onto obs times
        print("Interpolating in time...", flush=True)
        extracted = extracted.rename({"time": "t_dim"})
        extracted = extracted.interp(t_dim=ds.time.values, method=time_interp)

        # Put interp_dist into dataset
        extracted["interp_dist"] = interp_dist
        extracted = extracted.rename_vars({"t_dim": "time"})

        tg_out = Tidegauge()
        tg_out.dataset = extracted
        return tg_out
    
    
    def plot_on_map(self):
        """
        Show the location of a tidegauge on a map.

        Example usage:
        --------------
        # For a TIDEGAUGE object tg
        tg.plot_map()

        """

        debug(f"Plotting tide gauge locations for {get_slug(self)}")

        X = self.dataset.longitude
        Y = self.dataset.latitude
        fig, ax = plot_util.geo_scatter(X, Y)
        ax.set_xlim((X.min() - 10, X.max() + 10))
        ax.set_ylim((Y.min() - 10, Y.max() + 10))
        return fig, ax
