from .COAsT import COAsT
from warnings import warn
import numpy as np

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
        """ A method to make accessing the following simpler
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
