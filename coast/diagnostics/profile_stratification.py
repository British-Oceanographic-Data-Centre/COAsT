from ..data.profile import Profile
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import copy
from .._utils.plot_util import geo_scatter
from .._utils.logging_util import get_slug, debug


class ProfileStratification(Profile):  # TODO All abstract methods should be implemented
    """
    Object for handling and storing necessary information, methods and outputs
    for calculation of stratification diagnostics.

    Related to GriddedStratification class

    Parameters
    ----------
        profile : xr.Dataset
            Profile object on assumed on t-points.
    """

    def __init__(self, profile: xr.Dataset):
        # TODO Super __init__ should be called at some point
        debug(f"Creating new {get_slug(self)}")
        self.dataset = xr.Dataset()

        # Define the dimensional sizes as constants
        self.nid = profile.dataset.dims["id_dim"]
        self.nz = profile.dataset.dims["z_dim"]
        debug(f"Initialised {get_slug(self)}")

    def calc_pea(self, profile: xr.Dataset, Zmax):
        """
        Calculates Potential Energy Anomaly

        The density and depth averaged density can be supplied within profile as "density" and
        "density_bar" DataArrays, respectively. If they are not supplied they will be calculated.
        "density_bar" is calculated using depth averages of temperature and salinity.

        Writes self.dataset.pea
        """
        # may be duplicated in other branches. Uses the integral of T&S rather than integral of rho approach
        gravity = 9.81

        # Define grid spacing, dz. Required for depth integral
        profile.calculate_vertical_spacing()
        dz = profile.dataset.dz

        # Depth, relabel for convenience
        depth_t = profile.dataset.depth

        # Construct a mask of zeros below threshold, floats above depth of Zmax threshold.
        # Floats are in the range (0,1] and represent the fractional proximity to Zmax.
        # Used for scaling layer thickness, which would then sum to Zmax.
        Zd_mask, kmax = profile.calculate_vertical_mask(Zmax)

        # Height is depth_t above Zmax. Except height is Zmax for the last level above Zmax.
        height = np.floor(Zd_mask) * depth_t + (np.ceil(Zd_mask) - np.floor(Zd_mask)) * Zmax

        if not "density" in profile.dataset:
            profile.construct_density(CT_AS=True, pot_dens=True)
        if not "density_bar" in profile.dataset:
            profile.construct_density(CT_AS=True, rhobar=True, Zd_mask=Zd_mask, pot_dens=True)
        rho = profile.dataset.variables["density"].fillna(0)  # density
        rhobar = profile.dataset.variables["density_bar"]  # density with depth-mean T and S

        pot_energy_anom = (height * (rho - rhobar) * dz).sum(dim="z_dim", skipna=True) * gravity / Zmax

        coords = {
            "time": ("id_dim", profile.dataset.time.values),
            "latitude": (("id_dim"), profile.dataset.latitude.values),
            "longitude": (("id_dim"), profile.dataset.longitude.values),
        }
        dims = ["id_dim"]
        attributes = {"units": "J / m^3", "standard_name": "Potential Energy Anomaly"}
        self.dataset["pea"] = xr.DataArray(pot_energy_anom, coords=coords, dims=dims, attrs=attributes)

    def quick_plot(self, var: xr.DataArray = None):
        """
        Map plot for potential energy anomaly.

        Parameters
        ----------
        var : xr.DataArray, optional
            Pass variable to plot. The default is None. In which case
            potential energy anomaly is plotted.

        Returns
        -------
        None.

        Example Usage
        -------------
        For a Profile object, profile
        pa = coast.ProfileStratification(profile)
        pa.calc_pea(profile, 200)
        pa.quick_plot( 'pea' )
        """

        debug(f"Generating quick plot for {get_slug(self)}")

        if var is None:
            var_lst = [self.dataset.pea]
        else:
            var_lst = [self.dataset[var]]

        fig = None
        ax = None
        for var in var_lst:

            title_str = var.attrs["standard_name"] + " (" + var.attrs["units"] + ")"

            fig, ax = geo_scatter(
                self.dataset.longitude,
                self.dataset.latitude,
                var,
                title=title_str,
            )

        return fig, ax
