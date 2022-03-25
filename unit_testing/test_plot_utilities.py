"""

"""

# IMPORT modules. Must have unittest, and probably coast.
from coast import plot_util
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import os


class test_plot_utilities(unittest.TestCase):
    def test_scatter_with_fit(self):
        x = np.arange(0, 50)
        y = np.arange(0, 50) / 1.5
        f, a = plot_util.scatter_with_fit(x, y)
        a.set_title("Test: Scatter_with_fit()")

        f.savefig(os.path.join(files.dn_fig, "scatter_with_fit_test.png"))
        plt.close("all")

    def test_geo_axes(self):
        lonbounds = [-20, 20]
        latbounds = [30, 60]
        f, a = plot_util.create_geo_axes(lonbounds, latbounds)
        a.set_title("Test: create_geo_axes()")
        a.scatter([0, -10], [50, 50])

        f.savefig(os.path.join(files.dn_fig, "create_geo_axes_test.png"))
        plt.close()

    def test_determine_colorbar_extension(self):
        pretend_data = np.arange(0, 50)
        test1 = plot_util.determine_colorbar_extension(pretend_data, -50, 100)
        test2 = plot_util.determine_colorbar_extension(pretend_data, 1, 100)
        test3 = plot_util.determine_colorbar_extension(pretend_data, -50, 48)
        test4 = plot_util.determine_colorbar_extension(pretend_data, 1, 48)

        # TEST: <description here>
        check1 = test1 == "neither"
        check2 = test2 == "min"
        check3 = test3 == "max"
        check4 = test4 == "both"

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
        self.assertTrue(check3, "check3")
        self.assertTrue(check4, "check4")

    def test_determine_clim_by_stdev(self):
        pretend_data = np.arange(0, 100)
        pretend_data[-1] = 200
        clim = plot_util.determine_clim_by_standard_deviation(pretend_data, n_std_dev=2)

        # TEST: <description here>
        check1 = clim[0] == -13.808889915793792
        check2 = clim[1] == 114.82888991579378
        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
