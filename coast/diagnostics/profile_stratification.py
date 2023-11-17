from ..data.profile import Profile
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import copy
import coast
from .._utils.plot_util import geo_scatter
from .._utils.logging_util import get_slug, debug

####


####


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

    def clean_data(profile: xr.Dataset, gridded: xr.Dataset, Zmax):
        """
        Cleaning data for stratification metric calculations
        Stage 1:...

        stage 2...

        Stage 3. Fill gaps in data and extrapolate so there are T and S values where ever there is a depth value

        """
        # %%
        print("Cleaning the data")
        # find profiles good for SST and NBT
        dz_max = 25.0

        n_prf = profile.dataset.id_dim.shape[0]
        n_depth = profile.dataset.z_dim.shape[0]
        tmp_clean = profile.dataset.potential_temperature.values[:, :]
        sal_clean = profile.dataset.practical_salinity.values[:, :]

        any_tmp = np.sum(~np.isnan(tmp_clean), axis=1) != 0
        any_sal = np.sum(~np.isnan(sal_clean), axis=1) != 0

        # Find good SST and SSS depths
        def first_nonzero(arr, axis=0, invalid_val=np.nan):
            mask = arr != 0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

        if "bathymetry" in gridded.dataset:
            profile.gridded_to_profile_2d(gridded, "bathymetry")
            D_prf = profile.dataset.bathymetry.values
            z = profile.dataset.depth
            test_surface = z < np.minimum(dz_max, 0.25 * np.repeat(D_prf[:, np.newaxis], n_depth, axis=1))
            test_tmp = np.logical_and(test_surface, ~np.isnan(tmp_clean))
            test_sal = np.logical_and(test_surface, ~np.isnan(sal_clean))
            good_sst = np.zeros(n_prf) * np.nan
            good_sss = np.zeros(n_prf) * np.nan
            I_tmp = np.nonzero(np.any(test_tmp.values, axis=1))[0]
            I_sal = np.nonzero(np.any(test_sal.values, axis=1))[0]
            #
            # for ip in I_tmp:
            #    good_sst[ip] = np.min(np.nonzero(test_tmp.values[ip, :]))
            # for ip in I_sal:
            #    good_sss[ip] = np.min(np.nonzero(test_sal.values[ip, :]))

            good_sst = first_nonzero(test_tmp.values, axis=1)
            good_sss = first_nonzero(test_sal.values, axis=1)

            I_tmp = np.where(np.isfinite(good_sst))[0]
            I_sal = np.where(np.isfinite(good_sss))[0]

            #
            # find good profiles
            DD = np.minimum(Zmax, np.repeat(D_prf[:, np.newaxis], n_depth, axis=1))
            good_profile = np.array(np.ones(n_prf), dtype=bool)
            quart = [0, 0.25, 0.5, 0.75, 1]
            for iq in range(4):
                test = ~np.any(np.logical_and(z >= DD * quart[iq], z <= DD * quart[iq + 1]), axis=1)
                good_profile[test] = 0

            ###
        else:
            print("error no bathy provided, cant clean the data")
            return profile
        SST = np.zeros(n_prf) * np.nan
        SSS = np.zeros(n_prf) * np.nan

        SSS[I_sal] = sal_clean[I_sal, good_sss[I_sal].astype(int)]
        SST[I_tmp] = tmp_clean[I_tmp, good_sst[I_tmp].astype(int)]

        # fill holes in data
        # jth This is slow, there may be a more 'vector' way of doing it
        # %%
        for i_prf in range(n_prf):
            tmp = profile.dataset.potential_temperature.values[i_prf, :]
            sal = profile.dataset.practical_salinity.values[i_prf, :]
            z = profile.dataset.depth.values[i_prf, :]
            if any_tmp[i_prf]:
                tmp = coast.general_utils.fill_holes_1d(tmp)

                tmp[np.isnan(z)] = np.nan
                tmp_clean[i_prf, :] = tmp
            if any_sal[i_prf]:
                sal = coast.general_utils.fill_holes_1d(sal)

                sal[np.isnan(z)] = np.nan
                sal_clean[i_prf, :] = sal

        coords = {
            "time": ("id_dim", profile.dataset.time.values),
            "latitude": (("id_dim"), profile.dataset.latitude.values),
            "longitude": (("id_dim"), profile.dataset.longitude.values),
        }
        dims = ["id_dim", "z_dim"]
        profile.dataset["potential_temperature"] = xr.DataArray(tmp_clean, coords=coords, dims=dims)
        profile.dataset["practical_salinity"] = xr.DataArray(sal_clean, coords=coords, dims=dims)
        profile.dataset["sea_surface_temperature"] = xr.DataArray(SST, coords=coords, dims=["id_dim"])
        profile.dataset["sea_surface_salinity"] = xr.DataArray(SSS, coords=coords, dims=["id_dim"])
        profile.dataset["good_profile"] = xr.DataArray(good_profile, coords=coords, dims=["id_dim"])
        print("All nice and clean")
        # %%
        return profile

    def calc_pea(self, profile: xr.Dataset, gridded: xr.Dataset, Zmax):
        """
        Calculates Potential Energy Anomaly

        The density and depth averaged density can be supplied within profile as "density" and
        "density_bar" DataArrays, respectively. If they are not supplied they will be calculated.
        "density_bar" is calculated using depth averages of temperature and salinity.

        Writes self.dataset.pea
        """
        # may be duplicated in other branches. Uses the integral of T&S rather than integral of rho approach
        # %%
        gravity = 9.81
        # Clean data This is quit slow and over writes potential temperature and practical salinity variables
        profile = ProfileStratification.clean_data(profile, gridded, Zmax)

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
        # height = (
        #    np.floor(Zd_mask) * depth_t + (np.ceil(Zd_mask) - np.floor(Zd_mask)) * Zmax
        # )  # jth why not just use depth here?

        if not "density" in profile.dataset:
            profile.construct_density(CT_AS=False, pot_dens=True)
        if not "density_bar" in profile.dataset:
            profile.construct_density(CT_AS=False, rhobar=True, Zd_mask=Zd_mask, pot_dens=True)
        rho = profile.dataset.variables["density"].fillna(0)  # density
        rhobar = profile.dataset.variables["density_bar"]  # density with depth-mean T and S

        pot_energy_anom = (
            (depth_t * (rho - rhobar) * dz * Zd_mask).sum(dim="z_dim", skipna=True)
            * gravity
            / (dz * Zd_mask).sum(dim="z_dim", skipna=True)
        )
        # mask bad profiles
        pot_energy_anom = np.ma.masked_where(~profile.dataset.good_profile.values, pot_energy_anom.values)
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

    ##############################################################################
