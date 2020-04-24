from COAsT import COAsT
from warnings import warn


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
            self.bathy_metry = self.dataset.bath_metry
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
