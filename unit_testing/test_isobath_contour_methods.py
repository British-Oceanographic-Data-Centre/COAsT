"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime


class test_contour_f_methods(unittest.TestCase):
    def test_extract_isobath_contour_between_two_points(self):

        with self.subTest("Extract contour"):
            nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)
            contours, no_contours = coast.Contour.get_contours(nemo_f, 200)
            y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_f, contours[0], [50, -10], [60, 3])
            cont_f = coast.ContourF(nemo_f, y_ind, x_ind, 200)
            check1 = np.isclose(cont_f.y_ind.sum() + cont_f.y_ind.sum(), 190020)
            check2 = np.isclose(cont_f.data_contour.bathymetry.sum().item(), 69803.78125)

            self.assertTrue(check1, "check1")
            self.assertTrue(check2, "check2")

        with self.subTest("Plot contour"):
            coast.Contour.plot_contour(nemo_f, contour)
            cont_path = files.dn_fig + "contour.png"
            plt.savefig(cont_path)
            plt.close("all")

    def test_calculate_flow_across_contour(self):
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)
        nemo_u = coast.Gridded(
            fn_data=files.fn_nemo_grid_u_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_u_grid
        )
        nemo_v = coast.Gridded(
            fn_data=files.fn_nemo_grid_v_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_v_grid
        )
        contours, no_contours = coast.Contour.get_contours(nemo_f, 200)
        y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_f, contours[0], [50, -10], [60, 3])
        cont_f = coast.ContourF(nemo_f, y_ind, x_ind, 200)
        cont_f.calc_cross_contour_flow(nemo_u, nemo_v)
        check1 = np.allclose(
            (cont_f.data_cross_flow.normal_velocities + cont_f.data_cross_flow.depth_integrated_normal_transport).sum(),
            -1152.3771,
        )
        self.assertTrue(check1, "check1")

    def test_calculate_pressure_gradient_driven_flow(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)
        contours, no_contours = coast.Contour.get_contours(nemo_f, 200)
        y_ind, x_ind, contour = coast.Contour.get_contour_segment(nemo_f, contours[0], [50, -10], [60, 3])
        cont_f = coast.ContourF(nemo_f, y_ind, x_ind, 200)
        cont_f.calc_geostrophic_flow(
            nemo_t, config_u=files.fn_config_u_grid, config_v=files.fn_config_v_grid, ref_density=1027
        )
        check1 = np.allclose(
            (
                cont_f.data_cross_flow.normal_velocity_hpg
                + cont_f.data_cross_flow.normal_velocity_spg
                + cont_f.data_cross_flow.transport_across_AB_hpg
                + cont_f.data_cross_flow.transport_across_AB_spg
            ).sum(),
            74.65002414,
        )
        self.assertTrue(check1, "check1")


class test_contour_t_methods(unittest.TestCase):
    def setUp(self):
        # This is called at the beginning of every test_ method.
        # load gridded data
        self.nemo_t = coast.Gridded(files.fn_nemo_grid_t_dat, files.fn_nemo_dom, config=files.fn_config_t_grid)
        # create contour dataset along 200 m isobath
        contours, no_contours = coast.Contour.get_contours(self.nemo_t, 200)
        y_ind, x_ind, contour = coast.Contour.get_contour_segment(self.nemo_t, contours[0], [50, -10], [60, 3])
        self.cont_t = coast.ContourT(self.nemo_t, y_ind, x_ind, 200)

    def test_along_contour_flow(self):
        # calculate flow along contour
        nemo_u = coast.Gridded(files.fn_nemo_grid_u_dat, files.fn_nemo_dom, config=files.fn_config_u_grid)
        nemo_v = coast.Gridded(files.fn_nemo_grid_v_dat, files.fn_nemo_dom, config=files.fn_config_v_grid)
        self.cont_t.calc_along_contour_flow(nemo_u, nemo_v)

        with self.subTest("Check on velocities"):
            cksum = (
                (
                    self.cont_t.data_along_flow.velocities
                    * self.cont_t.data_along_flow.e3
                    * self.cont_t.data_along_flow.e4
                )
                .sum()
                .values
            )
            self.assertTrue(
                np.isclose(cksum, 116660850), "velocities checksum: " + str(cksum) + ", should be 116660850"
            )

        with self.subTest("Check on transport"):
            cksum = (self.cont_t.data_along_flow.transport * self.cont_t.data_along_flow.e4).sum().values
            self.assertTrue(
                np.isclose(cksum, 116660850), "transports checksum: " + str(cksum) + ", should be 116660850"
            )

    def test_along_contour_2d_flow(self):
        # calculate flow along contour
        nemo_u = coast.Gridded(files.fn_nemo_grid_u_dat, files.fn_nemo_dom, config=files.fn_config_u_grid)
        nemo_v = coast.Gridded(files.fn_nemo_grid_v_dat, files.fn_nemo_dom, config=files.fn_config_v_grid)
        nemo_u.dataset = nemo_u.dataset.isel(z_dim=0).squeeze()
        nemo_v.dataset = nemo_v.dataset.isel(z_dim=0).squeeze()
        self.cont_t.calc_along_contour_flow_2d(nemo_u, nemo_v)

        cksum = (
            (self.cont_t.data_along_flow.velocities * self.cont_t.data_along_flow.e3_0 * self.cont_t.data_along_flow.e4)
            .sum()
            .values
        )
        self.assertTrue(np.isclose(cksum, 293910.94), "velocities checksum: " + str(cksum) + ", should be 293910.94")

    def test_calculate_pressure_along_contour(self):
        self.cont_t.construct_pressure(1027)
        check1 = np.allclose(
            (self.cont_t.data_contour.pressure_s + self.cont_t.data_contour.pressure_h_zlevels).sum().compute().item(),
            27490693.20181531,
        )
        self.assertTrue(check1)
