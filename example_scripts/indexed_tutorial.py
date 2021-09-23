"""
This is a demonstration script for using the INDEXED class in the COAsT
package. This object has strict data formatting requirements, which are
outlined for various (observational) data types.
"""

import coast 

dir = "/projectsa/IMMERSE/WP8/TestData/"

altimetry = coast.Altimetry(file_path='./example_files/COAsT_example_altimetry_data.nc',config='./config/example_altimetry.json')

# Inspect object
#$ altimetry.dataset

profile = coast.Profile(file_path='./example_files/EN4_example.nc', config='./config/example_en4_profiles.json') 

# Inspect object
#$ profile.dataset

# GLider data https://linkedsystems.uk/erddap/files/Public_Glider_Data_0711/Doombar_20210214/Doombar_553_R.nc
# More gliders downloadable at https://www.bodc.ac.uk/data/bodc_database/gliders/
glider = coast.Glider(file_path=dir + "Bellatrix_555_R.nc", config='./config/example_glider_ego.json')

# Inspect object
#$ glider.dataset

argos = coast.Argos(file_path=dir + 'ARGOS.CSV',config='./config/example_argos.json')

# Inspect object
#$ argos.dataset

oceanparcels = coast.Oceanparcels(file_path= dir + 'OceanParcels_GFDL-ESM2M.2043r3.1.nc',config='./config/example_oceanparcels.json') 

# Inspect object
#$ oceanparcels.dataset

import datetime 
date0 = datetime.datetime(2007,1,10) 
date1 = datetime.datetime(2007,1,12) 

tidegauge = coast.Tidegauge(file_path='./example_files/tide_gauges/lowestoft-p024-uk-bodc', date_start = date0, date_end = date1, config='./config/example_tidegauge.json') 

# Inspect object
#$ tidegauge.dataset

