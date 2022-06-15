"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
from coast._utils import general_utils
import unittest
import numpy as np
import os.path as path
import xarray as xr
import matplotlib.pyplot as plt
import unit_test_files as files


class test_transect_methods(unittest.TestCase):
    def test_determine_extract_transect_indices(self):
        nemo_t = coast.Gridded(files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid)
        yt, xt, length_of_line = nemo_t.transect_indices([51, -5], [49, -9])

        # Test transect indices
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
        length_ref = 37

        check1 = xt == xt_ref
        check2 = yt == yt_ref
        check3 = length_of_line == length_ref
        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")
        self.assertTrue(check3, msg="check3")

    def test_calculate_transport_velocity_and_depth(self):
        with self.subTest("Calculate_transports and velocties and depth"):
            nemo_t = coast.Gridded(
                fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
            )
            nemo_u = coast.Gridded(
                fn_data=files.fn_nemo_grid_u_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_u_grid
            )
            nemo_v = coast.Gridded(
                fn_data=files.fn_nemo_grid_v_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_v_grid
            )
            nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)

            tran_f = coast.TransectF(nemo_f, (54, -15), (56, -12))
            tran_f.calc_flow_across_transect(nemo_u, nemo_v)
            cksum1 = tran_f.data_cross_tran_flow.normal_velocities.sum(dim=("t_dim", "z_dim", "r_dim")).item()
            cksum2 = tran_f.data_cross_tran_flow.normal_transports.sum(dim=("t_dim", "r_dim")).item()
            check1 = np.isclose(cksum1, -253.6484375)
            check2 = np.isclose(cksum2, -48.67562136873888)
            self.assertTrue(check1, msg="check1")
            self.assertTrue(check2, msg="check2")

        with self.subTest("plot_transect_on_map"):
            fig, ax = tran_f.plot_transect_on_map()
            ax.set_xlim([-20, 0])  # Problem: nice to make the land appear.
            ax.set_ylim([45, 65])  # But can not call plt.show() before adjustments are made...
            # fig.tight_layout()
            fig.savefig(files.dn_fig + "transect_map.png")
            plt.close("all")

        with self.subTest("plot_normal_velocity"):
            plot_dict = {"fig_size": (5, 3), "title": "Normal velocities"}
            fig, ax = tran_f.plot_normal_velocity(time=0, cmap="seismic", plot_info=plot_dict, smoothing_window=2)
            fig.tight_layout()
            fig.savefig(files.dn_fig + "transect_velocities.png")
            plt.close("all")

        with self.subTest("plot_depth_integrated_transport"):
            plot_dict = {"fig_size": (5, 3), "title": "Transport across AB"}
            fig, ax = tran_f.plot_depth_integrated_transport(time=0, plot_info=plot_dict, smoothing_window=2)
            fig.tight_layout()
            fig.savefig(files.dn_fig + "transect_transport.png")
            plt.close("all")

    def test_transect_density_and_pressure(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        tran_t = coast.TransectT(nemo_t, (54, -15), (56, -12))
        tran_t.construct_pressure()
        cksum1 = tran_t.data.density_zlevels.sum(dim=["t_dim", "r_dim", "depth_z_levels"]).compute().item()
        cksum2 = tran_t.data.pressure_h_zlevels.sum(dim=["t_dim", "r_dim", "depth_z_levels"]).compute().item()
        cksum3 = tran_t.data.pressure_s.sum(dim=["t_dim", "r_dim"]).compute().item()
        check1 = np.isclose(cksum1, 23800545.87457855)
        check2 = np.isclose(cksum2, 135536478.93335825)
        check3 = np.isclose(cksum3, -285918.5625)
        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")
        self.assertTrue(check3, msg="check3")

    def test_cross_transect_geostrophic_flow(self):
        nemo_f = coast.Gridded(fn_domain=files.fn_nemo_dom, config=files.fn_config_f_grid)
        tran_f = coast.TransectF(nemo_f, (54, -15), (56, -12))
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        tran_f.calc_geostrophic_flow(nemo_t, config_u=files.fn_config_u_grid, config_v=files.fn_config_v_grid)
        cksum1 = tran_f.data_cross_tran_flow.normal_velocity_hpg.sum(dim=("t_dim", "depth_z_levels", "r_dim")).item()
        cksum2 = tran_f.data_cross_tran_flow.normal_velocity_spg.sum(dim=("t_dim", "r_dim")).item()
        cksum3 = tran_f.data_cross_tran_flow.normal_transport_hpg.sum(dim=("t_dim", "r_dim")).item()
        cksum4 = tran_f.data_cross_tran_flow.normal_transport_spg.sum(dim=("t_dim", "r_dim")).item()

        check1 = np.isclose(cksum1, 84.8632969783)
        check2 = np.isclose(cksum2, -5.09718418121)
        check3 = np.isclose(cksum3, 115.2587369660)
        check4 = np.isclose(cksum4, -106.7897376093)

        self.assertTrue(check1, msg="check1")
        self.assertTrue(check2, msg="check2")
        self.assertTrue(check3, msg="check3")
        self.assertTrue(check4, msg="check4")
