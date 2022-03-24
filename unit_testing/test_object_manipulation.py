'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast import general_utils
import unittest
import numpy as np
import os.path as path
import xarray as xr
import unit_test_files as files

class test_object_manipulation(unittest.TestCase):
    
    def test_subset_single_variable(self):
        yt_ref = [
            164,
            163,
            162,
            162,
            161,
            160,
            159,
            158,
            157,
            156,
            156,
            155,
            154,
            153,
            152,
            152,
            151,
            150,
            149,
            148,
            147,
            146,
            146,
            145,
            144,
            143,
            142,
            142,
            141,
            140,
            139,
            138,
            137,
            136,
            136,
            135,
            134,
        ]
        xt_ref = [
            134,
            133,
            132,
            131,
            130,
            129,
            128,
            127,
            126,
            125,
            124,
            123,
            122,
            121,
            120,
            119,
            118,
            117,
            116,
            115,
            114,
            113,
            112,
            111,
            110,
            109,
            108,
            107,
            106,
            105,
            104,
            103,
            102,
            101,
            100,
            99,
            98,
        ]
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, 
                            config=files.fn_config_t_grid)
        data_t = sci.get_subset_as_xarray("temperature", xt_ref, yt_ref)

        # Test shape and exteme values
        check1 = (np.shape(data_t) == (51, 37))
        check2 =  (np.nanmin(data_t) - 11.267578 < 1e-6)
        check3 =  (np.nanmax(data_t) - 11.834961 < 1e-6)
    
        self.assertTrue(check1, 'check1')
        self.assertTrue(check2, 'check2')
        self.assertTrue(check3, 'check3')
        
    def test_indices_by_distance(self):
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, 
                            config=files.fn_config_t_grid)
        ind = sci.subset_indices_by_distance(0, 51, 111)

        # Test size of indices array
        check1 = np.shape(ind) == (2, 674)
        self.assertTrue(check1, 'check1')
        
    def test_interpolation_to_altimetry(self):
        
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, 
                            config=files.fn_config_t_grid)
        
        with self.subTest("Find nearest xy indices"):
            altimetry = coast.Altimetry(files.fn_altimetry)
            ind = altimetry.subset_indices_lonlat_box([-10, 10], [45, 60])
            altimetry_nwes = altimetry.isel(time=ind)  # nwes = northwest europe shelf
            ind_x, ind_y = general_utils.nearest_indices_2d(
                sci.dataset.longitude, sci.dataset.latitude, 
                altimetry_nwes.dataset.longitude, altimetry_nwes.dataset.latitude
            )
            check1 = ind_x.shape == altimetry_nwes.dataset.longitude.shape
            self.assertTrue(check1, 'check1')
        
        with self.subTest("Interpolate in space"):
            interp_lon = np.array(altimetry_nwes.dataset.longitude).flatten()
            interp_lat = np.array(altimetry_nwes.dataset.latitude).flatten()
            interpolated = sci.interpolate_in_space(sci.dataset.ssh, 
                                                    interp_lon, interp_lat)
        
            # Check that output array longitude has same shape as altimetry
            check1 = interpolated.longitude.shape == altimetry_nwes.dataset.longitude.shape
            self.assertTrue(check1, 'check1')
            
        with self.subTest("Interpolate in time"):
            interpolated = sci.interpolate_in_time(interpolated, altimetry_nwes.dataset.time)

            # Check time in interpolated object has same shape
            check1 = interpolated.time.shape == altimetry_nwes.dataset.time.shape
            self.assertTrue(check1, 'check1')
        
        
    

