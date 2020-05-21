import coast
import numpy as np

fn_dom = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_NEMO_domain.nc'
fn_dat = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_NEMO_data.nc'
fn_alt = '/Users/Dave/Documents/Projects/WCSSP/Data/COAsT_example_altimetry_data.nc'

nemo_dom = coast.DOMAIN()
nemo_var = coast.NEMO()
alt_test = coast.ALTIMETRY()

nemo_dom.load(fn_dom)
nemo_var.load(fn_dat)
alt_test.load(fn_alt)

alt_test.set_command_variables()
nemo_var.set_command_variables()
nemo_dom.set_command_variables()

alt_test.extract_lonlat_box([-10,10], [45,65])
alt_test.extract_indices_all_var(np.arange(0,4))

crps_test = nemo_var.crps_sonf('ssh', nemo_dom, alt_test, 'sla_filtered',
                    nh_radius=1, nh_type = "box", cdf_type = "empirical",
                    time_interp = "nearest", plot=True)
