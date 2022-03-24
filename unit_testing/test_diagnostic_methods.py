'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import unit_test_files as files

class test_diagnostic_methods(unittest.TestCase):
    
    def test_compute_vertical_spatial_derivative(self):
        nemo_t = coast.Gridded(fn_data = files.fn_nemo_grid_t_dat, 
                               fn_domain = files.fn_nemo_dom, 
                               config = files.fn_config_t_grid)
        nemo_w = coast.Gridded(fn_domain=files.fn_nemo_dom, 
                               config=files.fn_config_w_grid)

        log_str = ""
        # Compute dT/dz
        nemo_w_1 = nemo_t.differentiate("temperature", 
                                        config_path=files.fn_config_w_grid, 
                                        dim="z_dim")
        if nemo_w_1 is None:  # Test whether object was returned
            log_str += "No object returned\n"
        # Make sure the hardwired grid requirements are present
        if not hasattr(nemo_w.dataset, "depth_0"):
            log_str += "Missing depth_0 variable\n"
        if not hasattr(nemo_w.dataset, "e3_0"):
            log_str += "Missing e3_0 variable\n"
        if not hasattr(nemo_w.dataset.depth_0, "units"):
            log_str += "Missing depth units\n"
        # Test attributes of derivative. This are generated last so can indicate earlier problems
        nemo_w_2 = nemo_t.differentiate("temperature", dim="z_dim", 
                                        out_var_str="dTdz", out_obj=nemo_w)
        if not nemo_w_2.dataset.dTdz.attrs == {"units": "degC/m", 
                                               "standard_name": "dTdz"}:
            log_str += "Did not write correct attributes\n"
        # Test auto-naming derivative. Again test expected attributes.
        nemo_w_3 = nemo_t.differentiate("temperature", dim="z_dim", 
                                        config_path=files.fn_config_w_grid)
        if not nemo_w_3.dataset.temperature_dz.attrs == {"units": "degC/m",
                                                         "standard_name": "temperature_dz"}:
            log_str += "Problem with auto-naming derivative field\n"

        ## Test numerical calculation. Differentiate f(z)=-z --> -1
        # Construct a depth variable - needs to be 4D
        nemo_t.dataset["depth4D"], _ = xr.broadcast(nemo_t.dataset["depth_0"], 
                                                    nemo_t.dataset["temperature"])
        nemo_w_4 = nemo_t.differentiate("depth4D", dim="z_dim", out_var_str="dzdz", 
                                        config_path=files.fn_config_w_grid)
        if not np.isclose(
            nemo_w_4.dataset.dzdz.isel(z_dim=slice(1, nemo_w_4.dataset.dzdz.sizes["z_dim"])).max(), -1
        ) or not np.isclose(nemo_w_4.dataset.dzdz.isel(z_dim=slice(1, nemo_w_4.dataset.dzdz.sizes["z_dim"])).min(), -1):
            log_str += "Problem with numerical derivative of f(z)=-z\n"

        check1 = log_str == ""
        self.assertTrue(check1, msg='check1')
        
    def test_contruct_density(self):
        nemo_t = coast.Gridded(fn_data=files.fn_nemo_grid_t_dat, 
                               fn_domain=files.fn_nemo_dom, 
                               config=files.fn_config_t_grid)
        nemo_t.construct_density()
        yt, xt, length_of_line = nemo_t.transect_indices([54, -15], [56, -12])

        check1 = np.allclose( nemo_t.dataset.density.sel(
            x_dim=xr.DataArray(xt, dims=["r_dim"]), 
            y_dim=xr.DataArray(yt, dims=["r_dim"])).sum(dim=["t_dim", "r_dim", "z_dim"])
                .item(),
                11185010.518671108,)
        self.assertTrue(check1, msg='check1')
        
    def test_construct_pycnocline_depth_and_thickness(self):
        nemo_t = None
        nemo_w = None
        nemo_t = coast.Gridded(files.fn_nemo_grid_t_dat_summer, 
                               files.fn_nemo_dom, config=files.fn_config_t_grid)
        # create an empty w-grid object, to store stratification
        nemo_w = coast.Gridded(fn_domain=files.fn_nemo_dom, 
                               config=files.fn_config_w_grid)

        with self.subTest("Construct pycnocline depth"):
            log_str = ""
            # initialise Internal Tide object
            IT = coast.InternalTide(nemo_t, nemo_w)
            if IT is None:  # Test whether object was returned
                log_str += "No object returned\n"
            # Construct pycnocline variables: depth and thickness
            IT.construct_pycnocline_vars(nemo_t, nemo_w)
    
            if not hasattr(nemo_t.dataset, "density"):
                log_str += "Did not create density variable\n"
            if not hasattr(nemo_w.dataset, "rho_dz"):
                log_str += "Did not create rho_dz variable\n"
    
            if not hasattr(IT.dataset, "strat_1st_mom"):
                log_str += "Missing strat_1st_mom variable\n"
            if not hasattr(IT.dataset, "strat_1st_mom_masked"):
                log_str += "Missing strat_1st_mom_masked variable\n"
            if not hasattr(IT.dataset, "strat_2nd_mom"):
                log_str += "Missing strat_2nd_mom variable\n"
            if not hasattr(IT.dataset, "strat_2nd_mom_masked"):
                log_str += "Missing strat_2nd_mom_masked variable\n"
            if not hasattr(IT.dataset, "mask"):
                log_str += "Missing mask variable\n"
    
            # Check the calculations are as expected
            check1 = np.isclose(IT.dataset.strat_1st_mom.sum(), 3.74214231e08)
            check2 = np.isclose(IT.dataset.strat_2nd_mom.sum(), 2.44203298e08)
            check3 = np.isclose(IT.dataset.mask.sum(), 450580)
            check4 =  np.isclose(IT.dataset.strat_1st_mom_masked.sum(), 3.71876949e08)
            check5 = np.isclose(IT.dataset.strat_2nd_mom_masked.sum(), 2.42926865e08)
    
            self.assertTrue(check1, msg=log_str)
            self.assertTrue(check2, msg=log_str)
            self.assertTrue(check3, msg=log_str)
            self.assertTrue(check4, msg=log_str)
            self.assertTrue(check5, msg=log_str)
        
        with self.subTest("Plot pycnocline depth"):
            fig, ax = IT.quick_plot("strat_1st_mom_masked")
            fig.tight_layout()
            fig.savefig(files.dn_fig + "strat_1st_mom.png")
            plt.close('all')
        
        
