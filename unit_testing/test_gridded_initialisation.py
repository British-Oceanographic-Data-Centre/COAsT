"""Testing initialisation of gridded objects."""

# IMPORT modules. Must have unittest, and probably coast.
import unittest

import numpy as np
import xarray as xr

import unit_test_files as files
import coast


class test_gridded_initialisation(unittest.TestCase):
    def test_gridded_load_of_data_and_domain(self):
        # Check successfully load example data and domain
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        sci_attrs_ref = dict(
            [
                ("name", "AMM7_1d_20070101_20070131_25hourm_grid_T"),
                ("description", "ocean T grid variables, 25h meaned"),
                ("title", "ocean T grid variables, 25h meaned"),
                ("Conventions", "CF-1.6"),
                ("timeStamp", "2019-Dec-26 04:35:28 GMT"),
                ("uuid", "96cae459-d3a1-4f4f-b82b-9259179f95f7"),
            ]
        )

        # checking is LHS is a subset of RHS
        check1 = sci_attrs_ref.items() <= sci.dataset.attrs.items()

        self.assertTrue(check1, msg="Check1")

    def test_gridded_load_of_data_only(self):
        # Check load only data
        ds = xr.open_dataset(files.fn_nemo_dat)
        sci_load_ds = coast.Gridded(config=files.fn_config_t_grid)
        sci_load_ds.load_dataset(ds)
        sci_load_file = coast.Gridded(config=files.fn_config_t_grid)
        sci_load_file.load(files.fn_nemo_dat)
        check1 = sci_load_ds.dataset.identical(sci_load_file.dataset)
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_variables_correctly_renamed(self):
        # Check temperature is correctly renamed
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        check1 = "temperature" in sci.dataset
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_dimensions_correctly_renamed(self):
        # Check gridded dimensions are correctly renamed
        sci = coast.Gridded(files.fn_nemo_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        check1 = sci.dataset.temperature.dims == ("t_dim", "z_dim", "y_dim", "x_dim")
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_domain_only(self):
        # Check gridded load domain only
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)

        check1 = False
        if nemo_f.dataset._coord_names == {"depth_0", "latitude", "longitude"}:
            var_name_list = []
            for var_name in nemo_f.dataset.data_vars:
                var_name_list.append(var_name)
            if var_name_list == ["bathy_metry", "e1", "e2", "e3_0"]:
                check1 = True
        self.assertTrue(check1, msg="check1")

    def test_gridded_calculate_depth0_for_tuvwf(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        if not np.isclose(np.nansum(nemo_t.dataset.depth_0.values), 1705804300.0):
            raise ValueError(" X - Gridded depth_0 failed on t-grid failed")
        nemo_u = coast.Gridded(
            fn_data=files.fn_nemo_grid_u_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_u_grid
        )
        if not np.isclose(np.nansum(nemo_u.dataset.depth_0.values), 1705317600.0):
            raise ValueError(" X - Gridded depth_0 failed on u-grid failed")
        nemo_v = coast.Gridded(
            fn_data=files.fn_nemo_grid_v_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_v_grid
        )
        if not np.isclose(np.nansum(nemo_v.dataset.depth_0.values), 1705419100.0):
            raise ValueError(" X - Gridded depth_0 failed on v-grid failed")
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)
        if not np.isclose(np.nansum(nemo_f.dataset.depth_0.values), 1704932600.0):
            raise ValueError(" X - Gridded depth_0 failed on f-grid failed")

    # we seem to be missing a w-grid from tuvwf
    def test_gridded_calculate_bathymetry_for_t(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat,
            fn_domain=files.fn_nemo_dom,
            config=files.fn_config_t_grid,
            calculate_bathymetry=True,
        )
        nemo_t.make_lonLat_2d()
        if not np.isclose(np.nansum(nemo_t.dataset.bathy_metry.values), 116707590.0):
            raise ValueError(" X - Gridded calc_bathy failed on t-grid failed")

    # def test_gridded_calculate_bathymetry_for_u(self):
    #     nemo_u = coast.Gridded(
    #         fn_data=files.fn_nemo_grid_u_dat,
    #         fn_domain=files.fn_nemo_dom,
    #         config=files.fn_config_u_grid,
    #         calculate_bathymetry=True,
    #     )
    #     nemo_u.make_lonLat_2d()
    #     if not np.isclose(np.nansum(nemo_u.dataset.bathy_metry.values), 116031920.0):
    #         raise ValueError(" X - Gridded calc_bathy failed on u-grid failed")

    # def test_gridded_calculate_bathymetry_for_v(self):
    #     nemo_v = coast.Gridded(
    #         fn_data=files.fn_nemo_grid_v_dat,
    #         fn_domain=files.fn_nemo_dom,
    #         config=files.fn_config_v_grid,
    #         calculate_bathymetry=True,
    #     )
    #     nemo_v.make_lonLat_2d()
    #     if not np.isclose(np.nansum(nemo_v.dataset.bathy_metry.values), 116164800.0):
    #         raise ValueError(" X - Gridded calc_bathy failed on v-grid failed")

    # def test_gridded_calculate_bathymetry_for_f(self):
    #     nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid, calculate_bathymetry=True)
    #     nemo_f.make_lonLat_2d()
    #     if not np.isclose(np.nansum(nemo_f.dataset.bathy_metry.values), 115460610.0):
    #         raise ValueError(" X - Gridded calc_bathy failed on f-grid failed")

    def test_gridded_load_subregion_with_domain(self):
        amm7 = coast.Gridded(files.fn_nemo_dat_subset, files.fn_nemo_dom, config=files.fn_config_t_grid)

        # checking all the coordinates mapped correctly to the dataset object
        check1 = amm7.dataset._coord_names == {"depth_0", "latitude", "longitude", "time"}
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_multiple(self):
        amm7 = coast.Gridded(files.file_names_amm7, files.fn_nemo_dom, config=files.fn_config_t_grid, multiple=True)

        # checking all the coordinates mapped correctly to the dataset object
        check1 = amm7.dataset.time.size == 14
        self.assertTrue(check1, msg="check1")

    def test_gridded_compute_e3_from_ssh(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )

        e3t, e3u, e3v, e3f, e3w = coast.Gridded.get_e3_from_ssh(nemo_t, True, True, True, True, True)
        cksum = np.array([e3t.sum(), e3u.sum(), e3v.sum(), e3f.sum(), e3w.sum()])
        # these references are based on the example file's ssh field
        reference = np.array([8.337016e08, 8.333972e08, 8.344886e08, 8.330722e08, 8.265948e08])
        check1 = np.allclose(cksum, reference)
        self.assertTrue(check1, msg="check1")
