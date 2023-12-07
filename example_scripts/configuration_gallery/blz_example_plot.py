"""
blz_example_plot.py

Make simple Belize SSH plot.

"""

# %%
import coast
import matplotlib.pyplot as plt

#################################################
# %%  Loading  data
#################################################


dir_nam = "/projectsa/accord/GCOMS1k/OUTPUTS/BLZE12_02/2015/"
fil_nam = "BLZE12_1h_20151101_20151130_grid_T.nc"
dom_nam = "/projectsa/accord/GCOMS1k/INPUTS/BLZE12_C1/domain_cfg.nc"
config_t = "https://raw.githubusercontent.com/British-Oceanographic-Data-Centre/COAsT/master/config/example_nemo_grid_t.json"
config_u = "https://raw.githubusercontent.com/British-Oceanographic-Data-Centre/COAsT/master/config/example_nemo_grid_u.json"
config_v = "https://raw.githubusercontent.com/British-Oceanographic-Data-Centre/COAsT/master/config/example_nemo_grid_v.json"
config_w = "https://raw.githubusercontent.com/British-Oceanographic-Data-Centre/COAsT/master/config/example_nemo_grid_w.json"

sci_t = coast.Gridded(dir_nam + fil_nam, dom_nam, config=config_t)

sci_u = coast.Gridded(dir_nam + fil_nam.replace("grid_T", "grid_U"), dom_nam, config=config_u)

sci_v = coast.Gridded(dir_nam + fil_nam.replace("grid_T", "grid_V"), dom_nam, config=config_v)

# sci_v = coast.Nemo(dir_nam + fil_nam.replace("grid_T", "grid_V"), dom_nam, grid_ref="v-grid", multiple=False)

# create an empty w-grid object, to store stratification
sci_w = coast.Gridded(fn_domain=dom_nam, config=config_w)


# %% Plot
plt.pcolormesh(sci_t.dataset.ssh.isel(t_dim=0))
plt.show()
