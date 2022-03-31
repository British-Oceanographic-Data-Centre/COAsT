"""
seasia_r12_example_plot_bgc.py

Make simple SEAsia 1/12 deg DIC plot.

"""
#%%
import coast
import matplotlib.pyplot as plt


#################################################
#%%  Loading  data
#################################################
path_examples = "/projectsa/COAsT/NEMO_example_data/SEAsia_R12/"  ## data local

fn_seasia_domain = path_examples + "coast_example_domain_SEAsia.nc"
fn_seasia_config_bgc = path_examples + "example_nemo_bgc.json"
fn_seasia_var = path_examples + "coast_example_SEAsia_BGC_1990.nc"

seasia_bgc = coast.Gridded(fn_data=fn_seasia_var, fn_domain=fn_seasia_domain, config=fn_seasia_config_bgc)

#%% Plot
fig = plt.figure()
plt.pcolormesh(
    seasia_bgc.dataset.longitude,
    seasia_bgc.dataset.latitude,
    seasia_bgc.dataset.DIC.isel(t_dim=0).isel(z_dim=0),
    cmap="RdYlBu_r",
    vmin=1600,
    vmax=2080,
)
plt.colorbar()
plt.title("DIC, mmol/m^3")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()
fig.savefig("seasia_DIC_surface.png")
