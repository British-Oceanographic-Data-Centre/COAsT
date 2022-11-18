"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime


class test_profile_methods(unittest.TestCase):
    def test_load_process_and_compare_profile_data(self):

        with self.subTest("Load profile data from EN4"):
            profile = coast.Profile(config=files.fn_profile_config)
            profile.read_en4(files.fn_profile)
            profile.dataset = profile.dataset.isel(id_dim=np.arange(0, profile.dataset.dims["id_dim"], 10)).load()

            check1 = type(profile) == coast.Profile
            check2 = np.isclose(profile.dataset.temperature.values[0, 0], 8.981)

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Process EN4 profile data"):
            processed = profile.process_en4()
            processed.dataset.load()

            check1 = type(processed) == coast.Profile
            check2 = np.isnan(processed.dataset.temperature.values[0, 0])
            check3 = processed.dataset.dims["id_dim"] == 111

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")
            self.assertTrue(check3, "check3")

        with self.subTest("Compute vertical spacing"):
            profile.calculate_vertical_spacing()
            check1 = np.allclose(
                profile.dataset.dz.sum(dim="z_dim").isel(id_dim=[5, 10, 15]).values,
                np.array([1949.1846, 1972.8088, 21.5]),
            )
            self.assertTrue(check1, "check1")

    def test_compute_density(self):
        profile = coast.Profile(config=files.fn_profile_config)
        profile.read_en4(files.fn_profile)
        profile.dataset = profile.dataset.isel(id_dim=np.arange(0, profile.dataset.dims["id_dim"], 10)).load()

        profile.construct_density()

        check1 = np.allclose(
            profile.dataset.density.sum(dim=["id_dim", "z_dim"]).item(),
            4248551.199925806,
        )
        # Density depth mean T and S limited to 200m
        Zmax = 200  # m
        Zd_mask, kmax = profile.calculate_vertical_mask(Zmax)
        profile.construct_density(rhobar=True, pot_dens=True, CT_AS=True, Zd_mask=Zd_mask)
        check2 = np.allclose(profile.dataset.density_bar.mean(dim=["id_dim", "z_dim"]).item(), 1023.211151279021)
        # Temperature component of density (ie from depth mean Sal). full depth
        profile.construct_density(rhobar=True, pot_dens=True, CT_AS=True, Tbar=False)
        check3 = np.allclose(profile.dataset.density_T.mean(dim=["id_dim", "z_dim"]).item(), 1026.749192955557)
        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")
        self.assertTrue(check3, msg="check3")

    def test_compare_processed_profile_with_model(self):

        profile = coast.Profile(config=files.fn_profile_config)
        profile.read_en4(files.fn_profile)
        profile.dataset = profile.dataset.isel(id_dim=np.arange(0, profile.dataset.dims["id_dim"], 10)).load()
        processed = profile.process_en4()
        processed.dataset.load()

        with self.subTest("Gridded obs_operator"):
            nemo_t = coast.Gridded(
                fn_data=files.fn_nemo_grid_t_dat,
                fn_domain=files.fn_nemo_dom,
                config=files.fn_config_t_grid,
                calculate_bathymetry=True,
            )
            nemo_t.dataset["landmask"] = nemo_t.dataset.bottom_level == 0
            nemo_profiles = processed.obs_operator(nemo_t)

            check1 = type(nemo_profiles) == coast.Profile
            check2 = "nearest_index_x" in list(nemo_profiles.dataset.keys())
            check3 = np.isclose(nemo_profiles.dataset.interp_dist.values[0], 151.4443554515237)

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")
            self.assertTrue(check3, "check3")

        with self.subTest("Vertical interpolation"):
            pa = coast.ProfileAnalysis()
            reference_depths = np.arange(0, 500, 2)
            nemo_profiles.dataset = nemo_profiles.dataset.rename({"depth_0": "depth"})
            model_interpolated = pa.interpolate_vertical(nemo_profiles, processed)

            check1 = type(model_interpolated) == coast.Profile
            check2 = np.isclose(nemo_profiles.dataset.temperature.values[0, 0], np.float32(1.7324219))
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Vertical interpolation with NAN profiles"):
            nemo_profiles2 = coast.Profile(dataset=nemo_profiles.dataset.copy())
            nemo_profiles2.dataset.temperature[:10, :] = np.nan
            model_interpolated = pa.interpolate_vertical(nemo_profiles2, processed)

            check1 = type(model_interpolated) == coast.Profile
            check2 = np.isnan(nemo_profiles.dataset.temperature.values[:10, :]).all()
            check3 = ~np.isnan(nemo_profiles.dataset.temperature.values[11, 0])
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Profile Differencing"):
            difference = pa.difference(processed, model_interpolated)
            difference.dataset.load()

            check1 = type(difference) == coast.Profile
            check2 = np.isclose(difference.dataset.diff_temperature.values[22, 2], 2.2567890882492065)
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Regional Averaging"):
            mm = coast.MaskMaker()

            # Make some variables easier to access
            bath = nemo_t.dataset.bathymetry.values
            lon = nemo_t.dataset.longitude.values
            lat = nemo_t.dataset.latitude.values

            mm_north_sea = mm.region_def_nws_north_sea(lon, lat, bath)
            mm_whole_domain = np.ones(lon.shape)
            mask_list = [mm_north_sea, mm_whole_domain]
            mask_names = ["North Sea", "Whole Domain"]

            # Turn mask list into an xarray dataset
            mask_list = coast.MaskMaker.make_mask_dataset(lon, lat, mask_list)

            # Determine whether each profile is in each masked region or not
            mask_indices = pa.determine_mask_indices(model_interpolated, mask_list)

            # Do average differences for each region
            mask_means = pa.mask_means(difference, mask_indices)

            check1 = np.isclose(mask_means.all_mean_diff_temperature.values[0], -0.8169331445866469)
            self.assertTrue(check1, "check1")

        with self.subTest("Surface/Bottom averaging"):
            surface = 5
            model_profiles_surface = pa.depth_means(nemo_profiles, [0, surface])

            # Lets get bottom values by averaging over the bottom 30m, except whether
            # depth is <100m, then average over the bottom 10m
            model_profiles_bottom = pa.bottom_means(nemo_profiles, [10, 30], [100, np.inf])

            check1 = type(model_profiles_surface) == coast.Profile
            check2 = type(model_profiles_bottom) == coast.Profile
            check3 = np.isclose(model_profiles_surface.dataset.temperature.values[11], 6.6600165)
            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")
            self.assertTrue(check3, "check3")

    def test_calculate_vertical_mask(self):
        # load example profile data
        profile = coast.Profile(config=files.fn_profile_config)
        profile.read_en4(files.fn_profile)
        profile.dataset = profile.dataset.isel(id_dim=slice(0, 3)).isel(z_dim=slice(0, 4))

        # Reassign values to depth, within a full profile object, to make it transparent
        arr = np.array([[1, 2, 3, np.nan], [15, 20, 25, 30], [4, 5, 15, np.nan]])
        depth = xr.DataArray(arr, dims=["id_dim", "z_dim"])
        profile.dataset["depth"] = depth

        mask, kmax = profile.calculate_vertical_mask(21)
        mask = mask.fillna(-999)

        check1 = (kmax == np.array([2, 1, 2])).all()
        check2 = (mask.values == np.array([[1.0, 1.0, 1.0, -999], [1.0, 0.8, 0.0, 0.0], [1.0, 1.0, 1.0, -999]])).all()

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
