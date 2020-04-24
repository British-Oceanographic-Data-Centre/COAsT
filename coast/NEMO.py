from COAsT import COAsT


class NEMO(COAsT):

    def __init__(self):
        super()
        self.ssh = None
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
            self.ssh = self.dataset.ssh
        except AttributeError as e:
            print(str(e))

        try:
            self.botpres = self.dataset.botpres
        except AttributeError as e:
            print(str(e))

        try:
            self.toce = self.dataset.toce
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
