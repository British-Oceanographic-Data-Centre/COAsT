'''

'''

# IMPORT modules. Must have unittest, and probably coast.
import unittest
import coast
import numpy as np
import unit_test_files as files
import os
import xarray as xr

class test_example_scripts(unittest.TestCase):
    
    def test_altimetry_tutorial(self):
        from example_scripts import altimetry_tutorial
        
    def test_tidegauge_tutorial(self):
        from example_scripts import tidegauge_tutorial
        
    def test_tidetable_tutorial(self):
        from example_scripts import tidetable_tutorial
        
    def test_export_to_netcdf_tutorial(self):
        from example_scripts import export_to_netcdf_tutorial
        
    def test_transect_tutorial(self):
        from example_scripts import transect_tutorial
        
    def test_contour_tutorial(self):
        from example_scripts import contour_tutorial
        
    #def test_internal_tide_pycnocline_diagnostics(self):
    #    import internal_tide_pycnocline_diagnostics