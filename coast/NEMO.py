from .COAsT import COAsT
import xarray as xa
import numpy as np
from dask import delayed, compute, visualize
import graphviz


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

    def subset_indices_by_distance(self, centre_lon, centre_lat, radius):
        '''
        This is just a sketch of what this type of routine might look like.
        It would read in model domain location information as well as user specified
        information on a point location: centre and radius (probably km). It
        goes on to calculate the distance between all model points and the
        specified point and compares these distances to the radius.
        '''

        # Flatten NEMO domain stuff.
        lat = self.dataset.nav_lat
        lon = self.dataset.nav_lon

        # Calculate the distances between every model point and the specified
        # centre. Calls another routine dist_haversine.

        nemo_dist = self.dist_haversine(centre_lon, centre_lat, lon, lat)

        # Reshape distance array back to original 2-dimensional form
        # nemo_dist = xa.DataArray(nemo_dist.data.reshape(self.dataset.nav_lat.shape), dims=['y', 'x'])

        # Get boolean array where the distance is less than the specified radius
        # using np.where
        nemo_indices_bool = nemo_dist < radius
        nemo_indices = np.where(nemo_indices_bool.compute())

        # Then these output tuples can be separated into x and y arrays if necessary.

        return nemo_indices

    def get_subset_of_var(self, var: str, points_x: slice, points_y: slice):
        # TODO this is most likely wrong
        smaller = self.dataset[var].isel(x=points_x, y=points_y)
        return smaller

    def get_contour_complex(self, var, points_x, points_y, points_z, tolerance: int = 0.2):
        smaller = self.dataset[var].sel(z=points_z, x=points_x, y=points_y, method='nearest', tolerance=tolerance)
        return smaller
