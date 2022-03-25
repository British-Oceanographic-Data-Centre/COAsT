"""

"""

# IMPORT modules. Must have unittest, and probably coast.
import coast
import unittest
import numpy as np
import matplotlib.pyplot as plt
import unit_test_files as files
import datetime


class test_eof_methods(unittest.TestCase):
    def test_compute_regular_eof(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        eofs = coast.compute_eofs(nemo_t.dataset.ssh)

        ssh_reconstruction = (eofs.EOF * eofs.temporal_proj).sum(dim="mode").sum(dim=["x_dim", "y_dim"])
        ssh_anom = (nemo_t.dataset.ssh - nemo_t.dataset.ssh.mean(dim="t_dim")).sum(dim=["x_dim", "y_dim"])

        # Check ssh anomaly is reconstructed at each time point
        check1 = np.allclose(ssh_reconstruction, ssh_anom, rtol=0.0001)
        var_cksum = eofs.variance.sum(dim="mode").compute().item()
        check2 = np.isclose(var_cksum, 100)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")

    def test_compute_heofs(self):
        nemo_t = coast.Gridded(
            fn_data=files.fn_nemo_grid_t_dat, fn_domain=files.fn_nemo_dom, config=files.fn_config_t_grid
        )
        heofs = coast.compute_hilbert_eofs(nemo_t.dataset.ssh)

        ssh_reconstruction = (
            (heofs.EOF_amp * heofs.temporal_amp * np.exp(1j * np.radians(heofs.EOF_phase + heofs.temporal_phase)))
            .sum(dim="mode")
            .real.sum(dim=["x_dim", "y_dim"])
        )

        ssh_anom = (nemo_t.dataset.ssh - nemo_t.dataset.ssh.mean(dim="t_dim")).sum(dim=["x_dim", "y_dim"])

        # Check ssh anomaly is reconstructed at each time point
        check1 = np.allclose(ssh_reconstruction, ssh_anom, rtol=0.0001)
        var_cksum = heofs.variance.sum(dim="mode").item()
        check2 = np.isclose(var_cksum, 100)

        self.assertTrue(check1, "check1")
        self.assertTrue(check2, "check2")
