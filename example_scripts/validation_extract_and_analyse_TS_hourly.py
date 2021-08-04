'''
This is an example script for doing a comparison of hourly NEMO temperature and salinity 
data against EN4 profile data.

The result of this script will be two analysis files. The first contains profile-by-profile
errors/CRPS at the surface and the bottom. The second contains the analysis in the first
averaged into user defined regional boxes.

IF you are dealing with a very large run and/or a very large domain
(>=20 years or a global, high resolution domain),
you may need to run the extract multiple times. Also, be careful with how
you chunk the NEMO data. It may be best to do some experimenting with this
option when creating your NEMO object. The extract step of this script can 
also be put into a parallel script. There is a parallel example script which
shows how this can be done with a slurm batch type setup.

Make sure to take a look at the docstrings for each routine, where there is 
more information.
'''

import sys
# IF USING A DEVELOPMENT BRANCH OF COAST, ADD THE REPOSITORY TO PATH:
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast
import xarray as xr
import numpy as np

# Name of the run
run_name='co7'

# File paths
fn_nemo_t = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/hourly/{0}/*.nc".format(run_name)
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
fn_en4 = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/en4/*.nc"
fn_extracted = "".format(run_name)
fn_regional = "".format(run_name)

# CREATE NEMO OBJECT
nemo = coast.NEMO(fn_nemo_t, fn_nemo_domain, multiple=True, 
                  chunks = {'time_counter': 100})
lon = nemo.dataset.longitude.values.squeeze()
lat = nemo.dataset.latitude.values.squeeze()

# CREATE EN4 PROFILE OBJECT
en4 = coast.PROFILE()
en4.read_EN4(fn_en4)

# IF you haven't yet processed EN4 data, you should do this now and save to
# a new file. Read this new file into a new PROFILE object with chunking
lonbounds = [np.nanmin(lon), np.nanmax(lon)]
latbounds = [np.nanmin(lat), np.nanmax(lat)]
_ = en4.process_en4(fn_en4, lonbounds, latbounds)

# Read processed data into profile object
en4.dataset = xr.open_dataset(fn_en4, chunks={'profile':10000})

# Use COAsT to make predefined REGIONAL MASKS.
# Alternatively, define your own in your own way, or read from file.
# Another alternative: don't make any masks, and the routine will only average
# over the whole domain
mm = coast.MASK_MAKER()
regional_masks = []
bath = nemo.dataset.bathymetry.load()
regional_masks.append( mm.region_def_nws_north_sea(lon,lat,bath.values))
regional_masks.append( mm.region_def_nws_outer_shelf(lon,lat,bath.values))
regional_masks.append( mm.region_def_english_channel(lon,lat,bath.values))
regional_masks.append( mm.region_def_nws_norwegian_trench(lon,lat,bath.values))
regional_masks.append( mm.region_def_kattegat(lon,lat,bath.values))
regional_masks.append( mm.region_def_south_north_sea(lon,lat,bath.values))
off_shelf = mm.region_def_off_shelf(lon, lat, bath.values)
off_shelf[regional_masks[3].astype(bool)] = 0
off_shelf[regional_masks[4].astype(bool)] = 0
regional_masks.append(off_shelf)
regional_masks.append(mm.region_def_irish_sea(lon, lat, bath.values))

region_names = ['north_sea','outer_shelf','eng_channel','nor_trench', 'kattegat', 'south_north_sea', 'off_shelf', 'irish_sea']

# EXTRACT and PROCESS model data at EN4 locations
# If you provide fn_extracted, the data will be saved to a new file
# The new 'extracted' instance, is a new PROFILE() object.
extracted = en4.extract_top_and_bottom(nemo, fn_extracted, surface_def=2, 
                                       bottom_def=10, do_crps=True, 
                                       crps_radii = [2,4,6])
										
# REGIONALLY AVERAGE errors using extracted data
extracted.analyse_top_and_bottom(fn_regional, regional_masks, region_names,
                                 dist_omit= 5)

# YOU will now have two files: one containing extracted model data at observation
# locations and times (nearest) along with some errors (differences) and another
# containing regional and climatological averages of these errors and CRPS.