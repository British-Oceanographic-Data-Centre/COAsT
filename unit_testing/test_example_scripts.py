"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import unittest
import matplotlib.pyplot as plt
from socket import gethostname


class test_example_scripts(unittest.TestCase):
    def test_altimetry_tutorial(self):
        from example_scripts import altimetry_tutorial

        plt.close("all")

    def test_tidegauge_tutorial(self):
        from example_scripts import tidegauge_tutorial

        plt.close("all")

    def test_tidetable_tutorial(self):
        from example_scripts import tidetable_tutorial

        plt.close("all")

    def test_export_to_netcdf_tutorial(self):
        from example_scripts import export_to_netcdf_tutorial

        plt.close("all")

    def test_transect_tutorial(self):
        from example_scripts import transect_tutorial

        plt.close("all")

    def test_contour_tutorial(self):
        from example_scripts import contour_tutorial

        plt.close("all")

    def test_internal_tide_pycnocline_diagnostics(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import internal_tide_pycnocline_diagnostics

            plt.close("all")

    def test_amm15_example_plot(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import amm15_example_plot

            plt.close("all")

    def test_anchor_plots_of_nsea_wvel(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import anchor_plots_of_nsea_wvel

            plt.close("all")

    def test_blz_example_plot(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import blz_example_plot

            plt.close("all")

    def test_seasia_r12_example_plot(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import seasia_r12_example_plot

            plt.close("all")

    def test_wcssp_india_example_plot(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import wcssp_india_example_plot

            plt.close("all")

    def test_internal_tide_pycnocline_diagnostics(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import internal_tide_pycnocline_diagnostics

            plt.close("all")

    def test_wod_bgc_ragged_example(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import wod_bgc_ragged_example

            plt.close("all")

    def test_seasia_dic_example_plot(self):
        if "livljobs" in gethostname().lower():
            from example_scripts import seasia_dic_example_plot

            plt.close("all")