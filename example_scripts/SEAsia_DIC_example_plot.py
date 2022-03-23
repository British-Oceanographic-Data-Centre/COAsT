"""
seasia_r12_example_plot_BGC.py

Make simple SEAsia 1/12 deg DIC plot.

"""
#%%
import coast
import matplotlib.pyplot as plt


#################################################
#%%  Loading  data
#################################################
path_examples = "/scratch/accord/COAST/coast_demo/COAsT_example_files/EXTRA_examples/"  ## data local

fn_SEAsia_domain = path_examples + "coast_example_domain_SEAsia.nc"
fn_SEAsia_config_BGC = path_examples + "example_nemo_BGC.json"
fn_SEAsia_var = path_examples + "coast_example_SEAsia_BGC_1990.nc"

SEAsia_BGC = coast.Gridded(fn_data=fn_SEAsia_var, fn_domain=fn_SEAsia_domain, config=fn_SEAsia_config_BGC)

#%% Plot
fig = plt.figure()
plt.pcolormesh(
    SEAsia_BGC.dataset.longitude,
    SEAsia_BGC.dataset.latitude,
    SEAsia_BGC.dataset.DIC.isel(t_dim=0).isel(z_dim=0),
    cmap="RdYlBu_r",
    vmin=1600,
    vmax=2080,
)
plt.colorbar()
plt.title("DIC, mmol/m^3")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()
fig.savefig("SEAsia_DIC_surface.png")
