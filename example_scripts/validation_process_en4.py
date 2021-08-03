"""
This is an example script for processing EN4 data prior to analysis. This should
be done before using any of the profile analysis validation tools when using
EN4 data. The processing will extract a domain of the data and apply QC flags.
The processed data will be saved to a new file, ready for use in the analysis
routines.
"""

import sys
# IF USING A DEVELOPMENT BRANCH OF COAST, ADD THE REPOSITORY TO PATH:
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast
import xarray as xr
import numpy as np

# 1) Define file paths for input and output (processed) EN4 data
#    fn_en4_in can be multiple files by using a glob (*).
#    fn_en4_out will be a single (concatenated) output file
fn_en4_in = "/gws/nopw/j04/jmmp/CO9_AMM15/obs/en4/*.nc"
fn_en4_out = "/gws/nopw/j04/jmmp/CO9_AMM15/obs/en4_processed_amm15.nc"

# 2) Define longitude and latitude box for which to extract EN4 data.
#    This should match your model domain at least approximately.
#    Any remaining points that lie outside your model domain can be omitted
#    in later routines.
lonbounds = [-25.46, 16.25]
latbounds = [44.06, 63.35]

# 3) Create COAsT PROFILE() object and read in EN4 data to its dataset
en4 = coast.PROFILE()
en4.read_EN4(fn_en4_in, multiple=True, chunks={'profile':10000})

# 4) Call the processing routine, which will automaticall write to file
en4.process_en4(fn_en4_out, lonbounds, latbounds)