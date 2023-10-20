import coast
import numpy as np
from os import path
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # colormap fiddling

# set some paths
root = "./"
dn_files = root + "./example_files/"
fn_prof = path.join(dn_files, "coast_example_en4_201008.nc")
fn_cfg_prof = path.join("config", "example_en4_profiles.json")

# Create a Profile object and load in the data:
profile = coast.Profile(config=fn_cfg_prof)
profile.read_en4(fn_prof)

processed_profile = profile.process_en4()
profile = processed_profile

pa = coast.ProfileStratification(profile)


fn_grd_dom = "example_files/coast_example_nemo_domain.nc"
fn_grd_cfg = "config/example_nemo_grid_t.json"
nemo = coast.Gridded(fn_domain=fn_grd_dom, config=fn_grd_cfg)
#profile.match_to_grid(nemo)
#profile.gridded_to_profile_2d(nemo, "bathymetry")

Zmax = 200  # metres
pa.calc_pea(profile, nemo, Zmax)
