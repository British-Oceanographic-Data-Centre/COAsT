'''
THIS is a sample script for setting up the loop for analysis of daily model means 
against EN4 profiles. This script should be modified to reflect local path/directory
structures and user preferences.

This script acts as a single process and can be used as part of a parallel analysis
(recommended). The command line input to this script is a single index, which refers
to the NEMO-EN4 file pair to analyse. File pairs are defined using datetime strings and
through the construction of file names using make_nemo_filename() and make_en4_filename().

Once everything is defined correctly, This script can be submitted to a SLURM-style
job-array. For example: 

   #!/bin/bash 
   #SBATCH --partition=short-serial 
   #SBATCH -o %A_%a.out
   #SBATCH -e %A_%a.err
   #SBATCH --time=30:00
   #SBATCH --array=1-132%10
   module add jaspy
   source ~/envs/coast/bin/activate
   python sample_script_mean_profile.py $SLURM_ARRAY_TASK_ID

Alternatively, it can be used as a template for serial analysis.

For more info, please see the Github:

https://github.com/JMMP-Group/NEMO_validation/wiki/T&S-%7C-Daily-Anomalies-with-Depth
'''

import numpy as np
import sys
import os

# CHANGE: SET THIS TO BE THE MODEL VALIDATION CODE DIRECTORY
os.chdir('/home/users/dbyrne/code/NEMO_validation/')

# CHANGE: IF USING A DEVELOPMENT VERSION OF COAST UNCOMMENT THIS LINE
sys.path.append('/home/users/dbyrne/code/COAsT')

import os.path
import coast 
from dateutil.relativedelta import *
from datetime import datetime

# CHANGE: START AND END DATES FOR ANALYSIS
start_date = datetime(2004,1,1) 
end_date = datetime(2014,12,1)

# CHANGE: DIRECTORIES AND FILE PATHS FOR ANALYSIS.
# NEMO data directory and domain file
dn_nemo_data = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/daily/p0/"
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
dn_en4 = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/en4/"
dn_out = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/analysis/tmp"
run_name = 'CO9p0 AMM15'

# ROUTINE FOR MAKING NEMO FILENAME FROM DATETIME OBJECT.
# CHANGE: MODIFY IF THIS DOESNT MATCH YOUR NAMING CONVENTION
def make_nemo_filename(dn, dt):
    ''' Make NEMO filename from directory name (dn) and datetime (dt)'''
    suffix = '_daily_grid_T'
    day = str(dt.day).zfill(2) 
    month = str(dt.month).zfill(2)
    year = dt.year
    yearmonth = str(year) + str(month) + day
    return os.path.join(dn, yearmonth + suffix + '.nc')

# ROUTINE FOR MAKING EN4 FILENAME FROM DATETIME OBJECT.
# CHANGE: MODIFY IF THIS DOESNT MATCH YOUR NAMING CONVENTION
def make_en4_filename(dn, dt):
    ''' Make EN4 filename from directory name (dn) and datetime (dt)'''
    prefix = 'EN.4.2.1.f.profiles.g10.'
    month = str(dt.month).zfill(2)
    year = dt.year
    yearmonth = str(year) + str(month)
    return os.path.join(dn, prefix + yearmonth + '.nc')

##################################################################

# Start of MAIN script

##################################################################

# Get input from command line
print(str(sys.argv[1]), flush=True)
index = int(sys.argv[1])
print(index)

n_months = (end_date.year - start_date.year)*12 + \
           (end_date.month - start_date.month) + 1
month_list = [start_date + relativedelta(months=+mm) for mm in range(0,n_months)]

nemo_filename = make_nemo_filename(dn_nemo_data, month_list[index])
en4_filename = make_en4_filename(dn_en4, month_list[index])

nemo = coast.NEMO(nemo_filename, fn_nemo_domain, chunks = {'time_counter':1})
en4 = coast.PROFILE()
en4.read_EN4(en4_filename)

fn_out = os.path.join(dn_out, os.path.splitext(os.path.basename(nemo_filename))[0] + '_out.nc')

en4.extract_profiles(nemo, fn_out, run_name)
