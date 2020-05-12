from .COAsT import COAsT
from warnings import warn
import numpy as np
import xarray as xa


class DOMAIN(COAsT):

    def __init__(self):
        super()
        self.bathy_metry = None
        self.nav_lat = None
        self.nav_lon = None
        self.e1u = None
        self.e1v = None
        self.e1t = None
        self.e1f = None
        self.e2u = None
        self.e2v = None
        self.e2t = None
        self.e2f = None

    def set_command_variables(self):
        """
         A method to make accessing the following simpler
                bathy_metry (t,y,x) - float - (m i.e. metres)
                nav_lat (y,x) - float - (deg)
                nav_lon (y,x) - float - (deg)
                e1u, e1v, e1t, e1f (t,y,x) - double - (m)
                e2u, e2v, e2t, e2f (t,y,x) - double - (m)
        """
        try:
            self.bathy_metry = self.dataset.bathy_metry
        except AttributeError as e:
            warn(str(e))

        try:
            self.nav_lat = self.dataset.nav_lat
        except AttributeError as e:
            warn(str(e))

        try:
            self.nav_lon = self.dataset.nav_lon
        except AttributeError as e:
            warn(str(e))

        try:
            self.e1u = self.dataset.e1u
        except AttributeError as e:
            warn(str(e))

        try:
            self.e1v = self.dataset.e1v
        except AttributeError as e:
            print(str(e))

        try:
            self.e1t = self.dataset.e1t
        except AttributeError as e:
            print(str(e))

        try:
            self.e1f = self.dataset.e1f
        except AttributeError as e:
            print(str(e))

        try:
            self.e2u = self.dataset.e2u
        except AttributeError as e:
            print(str(e))

        try:
            self.e2v = self.dataset.e2v
        except AttributeError as e:
            print(str(e))

        try:
            self.e2t = self.dataset.e2t
        except AttributeError as e:
            print(str(e))

        try:
            self.e2f = self.dataset.e2f
        except AttributeError as e:
            print(str(e))

    def subset_indices_by_distance(self, centre_lon, centre_lat, radius):
        """
        This is just a sketch of what this type of routine might look like.
        It would read in model domain location information as well as user specified
        information on a point location: centre and radius (probably km). It
        goes on to calculate the distance between all model points and the
        specified point and compares these distances to the radius.
        """

        # Flatten NEMO domain stuff.
        lat = self.dataset.nav_lat
        lon = self.dataset.nav_lon

        # Calculate the distances between every model point and the specified
        # centre. Calls another routine dist_haversine.

        dist = self.dist_haversine(centre_lon, centre_lat, lon, lat)

        # Reshape distance array back to original 2-dimensional form
        # nemo_dist = xa.DataArray(nemo_dist.data.reshape(self.dataset.nav_lat.shape), dims=['y', 'x'])

        # Get boolean array where the distance is less than the specified radius
        # using np.where
        indices_bool = dist < radius
        indices = np.where(indices_bool.compute())

        # Then these output tuples can be separated into x and y arrays if necessary.

        return indices

    def find_j_i(self, lat: int, lon: int, grid_ref: str):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12, t)

        :param lat: latitude
        :param lon: longitude
        :param grid_ref: the gphi/glam version a user wishes to search over
        :return: the y and x coordinates for the given grid_ref variable within the domain file
        """

        internal_lat = f"gphi{grid_ref}"
        internal_lon = f"glam{grid_ref}"
        dist2 = xa.ufuncs.square(self.dataset[internal_lat] - lat) + xa.ufuncs.square(self.dataset[internal_lon] - lon)
        [_, y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def transect_indices(self, start: tuple, end: tuple, grid_ref: str = 'T') -> tuple:
        """
        This methods returns the indices of a simple straight line transect.

        checks `grid_ref` has a value within (T, V, U, F) this corresponds to the gphi/glam variable a user wishes
        to use for looking up the indices from.

        :type start: tuple A lat/lon pair
        :type end: tuple A lat/lon pair
        :type grid_ref: str
        :return: array of y indices, array of x indices, number of indices in transect
        """

        assert isinstance(grid_ref, str) and grid_ref.upper() in ("T", "V", "U", "F"), \
            "grid_ref should be either \"T\", \"V\", \"U\", \"F\""


        letter = grid_ref.lower()

        [j1, i1] = self.find_j_i(start[0], start[1], letter)  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1], letter)  # lat , lon

        line_length = max(np.abs(j2 - j1), np.abs(i2 - i1))

        jj1 = [int(jj) for jj in np.round(np.linspace(j1, j2, num=line_length))]
        ii1 = [int(ii) for ii in np.round(np.linspace(i1, i2, num=line_length))]
        return jj1, ii1, line_length
