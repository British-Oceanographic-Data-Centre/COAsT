'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime

class test_profile_methods(unittest.TestCase):
    
    def test_load_process_and_compare_profile_data(self):
        
        with self.subTest("Load profile data from EN4"):
            profile = coast.Profile(files.fn_profile, 
                                    config=files.fn_profile_config)
            profile.dataset = profile.dataset.isel(profile=np.arange(0, 
                              profile.dataset.dims["profile"], 10)).load()
        
            check1 = type(profile) == coast.Profile
            check2 = profile.dataset.temperature.values[0, 0] = 8.981
            
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            
        with self.subTest("Process EN4 profile data"):
            processed = profile.process_en4()
            processed.dataset.load()
        
            check1 = type(processed) == coast.profile.Profile
            check2 = np.isnan(processed.dataset.temperature.values[0, 0])
            check3 = processed.dataset.dims["profile"] == 111
            
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            self.assertTrue(check3, 'check3')
            
    def test_compare_processed_profile_with_model(self):
        
        profile = coast.Profile(files.fn_profile, 
                                config=files.fn_profile_config)
        profile.dataset = profile.dataset.isel(profile=np.arange(0, 
                          profile.dataset.dims["profile"], 10)).load()
        processed = profile.process_en4()
        processed.dataset.load()
        
        with self.subTest("Gridded obs_operator"):
            nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, 
            config=files.fn_config_t_grid
            )
            nemo_t.dataset["landmask"] = nemo_t.dataset.bottom_level == 0
            nemo_profiles = processed.obs_operator(nemo_t)
        
            check1 = type(nemo_profiles) == coast.profile.Profile
            check2 = "nearest_index_x" in list(nemo_profiles.dataset.keys())
            check3 = np.isclose(nemo_profiles.dataset.interp_dist.values[0], 151.4443554515237)
            
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            self.assertTrue(check3, 'check3')
            
        with self.subTest("Vertical interpolation"):
            reference_depths = np.arange(0, 500, 2)
            nemo_profiles.dataset = nemo_profiles.dataset.rename({"depth_0": "depth"})
            model_interpolated = nemo_profiles.interpolate_vertical(processed)
        
            check1 = type(model_interpolated) == coast.profile.Profile
            check2 = np.isclose(nemo_profiles.dataset.temperature.values[0, 0], np.float32(1.7324219))
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            
        with self.subTest("Profile Differencing"):
            difference = processed.difference(model_interpolated)
            difference.dataset.load()
        
            check1 = type(difference) == coast.profile.Profile
            check2 = np.isclose(difference.dataset.diff_temperature.values[0, 2], np.float32(1.1402345))
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            
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
            mask_indices = model_interpolated.determine_mask_indices(mask_list)
             
            # Do average differences for each region
            mask_means = difference.mask_means(mask_indices)
             
            check1 = np.isclose(mask_means.average_diff_temperature.values[0], np.float32(-0.78869253))
            self.assertTrue(check1, 'check1')
            
        with self.subTest("Surface/Bottom averaging"):
            surface = 5
            model_profiles_surface = nemo_profiles.depth_means([0, surface])
        
            # Lets get bottom values by averaging over the bottom 30m, except whether
            # depth is <100m, then average over the bottom 10m
            model_profiles_bottom = nemo_profiles.bottom_means([10, 30], [100, np.inf])
        
            check1 = type(model_profiles_surface) == coast.profile.Profile
            check1 = type(model_profiles_bottom) == coast.profile.Profile
            check3 = np.isclose(model_profiles_surface.dataset.temperature.values[0], np.float32(1.7500391))
            self.assertTrue(check1, 'check1')
            self.assertTrue(check2, 'check2')
            self.assertTrue(check3, 'check3')
