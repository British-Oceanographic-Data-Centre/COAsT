"""
seasia_r12_example_plot.py

Make simple seasia 1/12 deg SSS plot.

"""

# %%
import coast
import matplotlib.pyplot as plt


#################################################
# %%  Loading  data
#################################################

dir_nam = "/projectsa/COAsT/NEMO_example_data/SEAsia_R12/"
fil_nam = "SEAsia_R12_5d_20120101_20121231_gridT.nc"
dom_nam = "domain_cfg_ORCA12_adj.nc"
config_t = (
    "https://raw.githubusercontent.com/British-Oceanographic-Data-Centre/COAsT/master/config/example_nemo_grid_t.json"
)

sci_t = coast.Gridded(dir_nam + fil_nam, dir_nam + dom_nam, config=config_t)

# %% Plot
fig = plt.figure()

plt.pcolormesh(sci_t.dataset.longitude, sci_t.dataset.latitude, sci_t.dataset.salinity.isel(t_dim=0).isel(z_dim=0))
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("SE Asia, surface salinity (psu)")
plt.colorbar()
plt.show()
fig.savefig("SEAsia_R12_SSS.png", dpi=120)
