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

# IF USING A DEVELOPMENT VERSION OF COAST UNCOMMENT THIS LINE
import sys
sys.path.append('/home/users/dbyrne/code/COAsT')
import coast 
import os
import os.path
from dateutil.relativedelta import *
from datetime import datetime

# 1) START AND END DATES FOR ANALYSIS. Even if you provide data that contains
# more dates, only those dates specified will be extracted
start_date = datetime(2004,1,1) 
end_date = datetime(2014,12,1)

# 2) Set full paths for input and output files
# fn = filename. dn = directory name
dn_nemo_data = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/outputs/daily/p0/"
fn_nemo_domain = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/inputs/CO7_EXACT_CFG_FILE.nc"
fn_en4 = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/obs/en4/en4_processed_amm15.nc"
dn_out = "/gws/nopw/j04/jmmp_collab/CO9_AMM15/analysis/tmp"

# 3) Define a name for the run being analysed.
run_name = 'CO9p0 AMM15'

# 4) Define a routine for making the NEMO filename based on the directory name
# (dn) and a datetime object (dt)
def make_nemo_filename(dn, dt):
    ''' Make NEMO filename from directory name (dn) and datetime (dt)'''
    suffix = '_daily_grid_T'
    day = str(dt.day).zfill(2) 
    month = str(dt.month).zfill(2)
    year = dt.year
    yearmonth = str(year) + str(month) + day
    return os.path.join(dn, yearmonth + suffix + '.nc')

# 5) Get input from command line. This integer is used to determine NEMO filename
print(str(sys.argv[1]), flush=True)
index = int(sys.argv[1])
print(index)

# 6) Make list of months based on start and end date
n_months = (end_date.year - start_date.year)*12 + \
           (end_date.month - start_date.month) + 1
month_list = [start_date + relativedelta(months=+mm) for mm in range(0,n_months)]

# 7) Make NEMO filename for this index
nemo_filename = make_nemo_filename(dn_nemo_data, month_list[index])

# 8) Make NEMO and EN4 PROFILE objects.
nemo = coast.NEMO(nemo_filename, fn_nemo_domain, chunks = {'time_counter':1})
en4 = coast.PROFILE()
en4.read_EN4(en4_filename, chunks={'N_PROF':10000})

# 9) Define output file as input file + _out
fn_out = os.path.join(dn_out, os.path.splitext(os.path.basename(nemo_filename))[0] + '_out.nc')

# 10) Run profile extraction routine
en4.extract_profiles(nemo, fn_out, run_name)
