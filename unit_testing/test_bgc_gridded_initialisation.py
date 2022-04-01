"""
TEST reading of wod profiles
"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import os.path as path
import xarray as xr

# IMPORT THIS TO HAVE ACCESS TO EXAMPLE FILE PATHS:
import unit_test_files as files

# Define a testing class. Absolutely fine to have one or multiple per file.
# Each class must inherit unittest.TestCase
class test_bgc_gridded_initialisation(unittest.TestCase):
    def test_gridded_load_bgc_data_and_domain(self):
        # Check successfully load example data and domain
        sci = coast.Gridded(files.fn_nemo_bgc, files.fn_nemo_dom_bgc, config=files.fn_nemo_config_bgc)
        sci_attrs_ref = dict(
            [
                ("name", "SEAsia_HAD_1m_19900101_19901231_ptrc_T"),
                ("description", "tracer variables"),
                ("title", "tracer variables"),
                ("Conventions", "CF-1.6"),
                ("timeStamp", "2020-Oct-07 10:11:58 GMT"),
                ("uuid", "701bb916-558d-4ee8-9cf6-89454c7bc99f"),
            ]
        )

        # checking is LHS is a subset of RHS
        check1 = sci_attrs_ref.items() <= sci.dataset.attrs.items()
        self.assertTrue(check1, msg="Check1")

    def test_gridded_load_bgc_data(self):
        # Check load only data
        ds = xr.open_dataset(files.fn_nemo_bgc)
        sci_load_ds = coast.Gridded(config=files.fn_nemo_config_bgc)
        sci_load_ds.load_dataset(ds)
        sci_load_file = coast.Gridded(config=files.fn_nemo_config_bgc)
        sci_load_file.load(files.fn_nemo_bgc)
        check1 = sci_load_ds.dataset.identical(sci_load_file.dataset)
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_bgc_variables_correctly_renamed(self):
        # Check temperature is correctly renamed
        sci = coast.Gridded(files.fn_nemo_bgc, files.fn_nemo_dom_bgc, config=files.fn_nemo_config_bgc)
        check1 = "dic" in sci.dataset
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_bgc_dimensions_correctly_renamed(self):
        # Check gridded dimensions are correctly renamed
        sci = coast.Gridded(files.fn_nemo_bgc, files.fn_nemo_dom_bgc, config=files.fn_nemo_config_bgc)
        check1 = sci.dataset.dic.dims == ("t_dim", "z_dim", "y_dim", "x_dim")
        self.assertTrue(check1, msg="check1")

    def test_gridded_load_bgc_domain_only(self):
        # Check gridded load domain only
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom_bgc, config=files.fn_nemo_config_bgc)

        check1 = False
        if nemo_f.dataset._coord_names == {"depth_0", "latitude", "longitude"}:
            var_name_list = []
            for var_name in nemo_f.dataset.data_vars:
                var_name_list.append(var_name)
            if var_name_list == ["bathymetry", "e1", "e2", "e3_0", "bottom_level"]:
                check1 = True
        self.assertTrue(check1, msg="check1")
