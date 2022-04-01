"""
Demonstration of wrapped longitude coordinates when finding the nearest
 grid index for specified latitude and lonitude
"""
import coast
import os.path as path

# Global data files
dn_files = "/projectsa/COAsT/GLOBAL_example_data/"
filename = "so_Omon_IPSL-CM5A-LR_historical_r1i1p1_200001-200512.nc"
fn_ispl_dat_t = path.join(dn_files, filename)
# Configuration files describing the data files
fn_config_t_grid = "./config/example_cmip5_grid_t.json"

# Load the data as a Gridded object
ispl = coast.Gridded(fn_ispl_dat_t, config=fn_config_t_grid)

[j1, i1] = ispl.find_j_i(-9, 50)
print(f"At (-9N,50E) nearest j,i indices: {j1,i1}")

[j1, i1] = ispl.find_j_i(-9, -310)  # Same point on globe gives same result
print(f"At (-9N,-310E) nearest j,i indices: {j1,i1}")
