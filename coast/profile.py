"""Profile Class"""
from .index import Indexed
import numpy as np
import xarray as xr
from . import general_utils, plot_util
import matplotlib.pyplot as plt
import glob
import datetime
from .logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
import xarray.ufuncs as uf


class Profile(Indexed):
    """
    OBSERVATION type class for storing data from a CTD Profile (or similar
    down and up observations). The structure of the class is based on data from
    the EN4 database. The class dataset should contain two dimensions:

        > profile :: The profiles dimension. Called N_PROF in EN4 data.
                     Each element of this dimension contains data for a
                     individual location.
        > level   :: The dimension for depth levels. Called N_LEVELS in EN4
                     files.
    """

    def __init__(self, file_path: str = None, multiple=False, config: Union[Path, str] = None):
        """Initialization and file reading.

        Args:
            file_path (str): path to data file
            multiple (boolean): True if reading multiple files otherwise False
            config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")
        super().__init__(config)

        if file_path is None:
            warn("Object created but no file or directory specified: \n" "{0}".format(str(self)), UserWarning)
        else:
            self.read_en4(file_path, self.chunks, multiple)
            self.apply_config_mappings()

        print(f"{get_slug(self)} initialised")

    def read_en4(self, fn_en4, chunks: dict = {}, multiple=False) -> None:
        """Reads a single or multiple EN4 netCDF files into the COAsT profile data structure.

        Args:
            fn_en4 (str): path to data file
            chunks (dict): chunks
            multiple (boolean): True if reading multiple files otherwise False
        """
        if not multiple:
            self.dataset = xr.open_dataset(fn_en4, chunks=chunks)
        else:
            if type(fn_en4) is not list:
                fn_en4 = [fn_en4]

            file_to_read = []
            for file in fn_en4:
                if "*" in file:
                    wildcard_list = glob.glob(file)
                    file_to_read = file_to_read + wildcard_list
                else:
                    file_to_read.append(file)

            # Reorder files to read
            file_to_read = np.array(file_to_read)
            dates = [ff[-9:-3] for ff in file_to_read]
            dates = [datetime.datetime(int(dd[0:4]), int(dd[4:6]), 1) for dd in dates]
            sort_ind = np.argsort(dates)
            file_to_read = file_to_read[sort_ind]

            for ff in range(0, len(file_to_read)):
                file = file_to_read[ff]
                data_tmp = xr.open_dataset(file, chunks=chunks)
                if ff == 0:
                    self.dataset = data_tmp
                else:
                    self.dataset = xr.concat((self.dataset, data_tmp), dim="N_PROF")

    """======================= Manipulate ======================="""

    def subset_indices_lonlat_box(self, lonbounds, latbounds):
        """Generates array indices for data which lies in a given lon/lat box.
        Keyword arguments:
        lon       -- Longitudes, 1D or 2D.
        lat       -- Latitudes, 1D or 2D
        lonbounds -- Array of form [min_longitude=-180, max_longitude=180]
        latbounds -- Array of form [min_latitude, max_latitude]

        return: Indices corresponding to datapoints inside specified box
        """
        ind = general_utils.subset_indices_lonlat_box(
            self.dataset.longitude, self.dataset.latitude, lonbounds[0], lonbounds[1], latbounds[0], latbounds[1]
        )
        return ind

    """======================= Plotting ======================="""

    def plot_profile(self, var: str, profile_indices=None):

        fig = plt.figure(figsize=(7, 10))

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims["profile"])
            pass

        for ii in profile_indices:
            prof_var = self.dataset[var].isel(profile=ii)
            prof_depth = self.dataset.depth.isel(profile=ii)
            ax = plt.plot(prof_var, prof_depth)

        plt.gca().invert_yaxis()
        plt.xlabel(var + "(" + self.dataset[var].units + ")")
        plt.ylabel("Depth (" + self.dataset.depth.units + ")")
        plt.grid()
        return fig, ax

    def plot_map(self, profile_indices=None, var_str=None, depth_index=None):

        if profile_indices is None:
            profile_indices = np.arange(0, self.dataset.dims["profile"])

        profiles = self.dataset.isel(profile=profile_indices)

        if var_str is None:
            fig, ax = plot_util.geo_scatter(profiles.longitude.values, profiles.latitude.values, s=5)
        else:
            print(profiles)
            c = profiles[var_str].isel(level=depth_index)
            fig, ax = plot_util.geo_scatter(profiles.longitude.values, profiles.latitude.values, c=c, s=5)
        return fig, ax

    def plot_ts_diagram(self, profile_index, var_t="potential_temperature", var_s="practical_salinity"):

        profile = self.dataset.isel(profile=profile_index)
        temperature = profile[var_t].values
        salinity = profile[var_s].values
        depth = profile.depth.values
        fig, ax = plot_util.ts_diagram(temperature, salinity, depth)

        return fig, ax

    """======================= Model Comparison ======================="""

        # Bottom values
 #       print('Averaging EN4 over bottom {0}m for bottom definition'.format(bottom_def), flush=True)
 #       bathy_pts = ds.bathymetry.isel(x_dim = ind2D[0], y_dim = ind2D[1]).swap_dims({'dim_0':'profile'})
 #       bottom_ind = en4.depth >= (bathy_pts - bottom_def)

#        sbt_en4 = en4[en4_tem_name].where(bottom_ind, np.nan)
#        sbs_en4 = en4[en4_sal_name].where(bottom_ind, np.nan)
#    
#        sbt_en4 = sbt_en4.mean(dim="z_dim", skipna=True).load()
#        sbs_en4 = sbs_en4.mean(dim="z_dim", skipna=True).load()

    def depth_means(self, depth_bounds):
        
        print('Averaging all variables between {0}m <= x < {1}m'.format(depth_bounds[0], depth_bounds[1]), flush=True)
        ds = self.dataset
        layer_ind0 = ds.depth >= depth_bounds[0]
        layer_ind1 = ds.depth < depth_bounds[1]
        layer_ind = layer_ind0 * layer_ind1
        masked = ds.where(layer_ind, np.nan)
        meaned = masked.mean(dim="z_dim", skipna=True).load()
        
        return meaned
    
    def bottom_means(self, ):
        return
    
    def determine_region_indices(self, masks):
        
        ds = self.dataset
        
        # If landmask not present, set to None for nearest_indices (no masking)
        if 'landmask' not in list(masks.keys()):
            landmask = None
        else:
            landmask = masks.landmask
        # SPATIAL indices - nearest neighbour
        ind_x, ind_y = general_utils.nearest_indices_2D(masks['longitude'], 
                                                        masks['latitude'],
                                                        ds['longitude'], 
                                                        ds['latitude'], 
                                                        mask=landmask)
        # Figure out which points lie in which region
        print('Figuring out which regions each profile is in..')
        region_indices = masks.isel(x_dim = ind_x, y_dim = ind_y)
        
        return region_indices.rename({'dim_0':'profile'})
        

    def mask_means(self, mask_indices, mask_names=None):
        
        ds = self.dataset
        
        n_masks = mask_indices.dims['dim_mask']
        
        if mask_names is None:
            mask_names = np.arange(n_masks)
        
        # Loop over maskal arrays. Assign mean to mask and seasonal means
        print('Calculating maskal averages..')
        for mm in range(0,n_masks):
            mask = mask_indices.isel(dim_mask = mm).mask.values
            mask_ind = np.where( mask.astype(bool) )[0]
            if len(mask_ind)<1:
                continue
            mask_data = ds.isel(profile = mask_ind)
            ds_average_prof = mask_data.mean(dim='profile', skipna=True).compute()
            ds_average_all = mask_data.mean(skipna=True).compute()
            
            var_list = list(ds_average_prof.keys())
            for vv in var_list:
                ds_average_prof = ds_average_prof.rename({vv:'profile_average_'+vv})
                ds_average_all = ds_average_all.rename({vv:'average_'+vv})
            
            ds_average_tmp = xr.merge((ds_average_prof, ds_average_all))
            
            if mm == 0:
                ds_average = ds_average_tmp
            else:
                ds_average = xr.concat((ds_average, ds_average_tmp), dim='dim_mask')
        
        return

    def difference(self, other, absolute_diff = True, square_diff = True):
        
        differenced = self.dataset - other.dataset
        diff_vars = list(differenced.keys())
        
        for vv in diff_vars:
            differenced = differenced.rename({vv:'diff_'+vv})
            
        if absolute_diff:
            abs_tmp = uf.fabs(differenced)
            diff_vars = list(abs_tmp.keys())
            for vv in diff_vars:
                abs_tmp = abs_tmp.rename({vv:'abs_'+vv})
        else:
            abs_tmp = xr.Dataset()
            
            
        if square_diff:
            sq_tmp = uf.square(differenced)
            diff_vars = list(sq_tmp.keys())
            for vv in diff_vars:
                sq_tmp = sq_tmp.rename({vv:'square_'+vv})
        else:
            sq_tmp = xr.Dataset()
            
        differenced = xr.merge((differenced, abs_tmp, sq_tmp))
        
        return_differenced = Profile()
        return_differenced.dataset = differenced
        
        return return_differenced

    def interpolate_vertical(self, new_depth, 
                             interp_method='linear'):
        
        if type(new_depth) is Profile:
            print('yesitis')
            new_depth = new_depth.dataset.depth
        
        ds = self.dataset
        n_prof = ds.dims['profile']
        
        # Now loop over profiles and interpolate model onto obs.
        print('Interpolating onto new depths...')
        count_ii=0
        for pp in range(0,n_prof):
            
            # Select the current profile
            profile = ds.isel(profile = pp).rename({'depth':'z_dim'})
            new_depth_prof = new_depth.isel(profile=pp).values
                    
            interpolated_tmp = profile.interp(z_dim=new_depth_prof, 
                                          method=interp_method)
            
            interpolated_tmp = interpolated_tmp.rename_vars({'z_dim':'depth'})
            interpolated_tmp = interpolated_tmp.reset_coords(['depth'])
            
            # If not first iteration, concat this interpolated profile
            if count_ii == 0:
                interpolated = interpolated_tmp
            else:
                interpolated = xr.concat((interpolated, interpolated_tmp),
                                           dim = 'profile', coords='all')
            count_ii = count_ii+1
        
        interpolated = interpolated.set_coords(['depth'])
        return_interpolated = Profile()
        return_interpolated.dataset = interpolated
            
        return return_interpolated

    def obs_operator(self, nemo, mask_bottom_level=True):
        '''
        VERSION 1.4 (05/07/2021)
        
        Extracts and does some basic analysis and identification of model data at obs
        locations, times and depths. Writes extracted data to file. This routine
        does the analysis per file, for example monthly files. Output can be
        subsequently input to analyse_ts_regional. Data is extracted saved in two
        forms: on the original model depth levels and interpolated onto the
        observed depth levels.
        
        This routine can be used in a loop to loop over all files containing
        daily or 25hourm data. Output files can be concatenated using a tools such
        as ncks or the concatenate_output_files() routine in this module.
        This is the recommended useage.
        
        INPUTS:
         nemo                 : NEMO object created on t-grid
         fn_out (str)         : Absolute filepath for desired output file.
                                If not provided, a delayed dataset will be returned
         run_name (str)       : Name of run. [default='Undefined']
                                Will be saved to output
         z_interp (str)       : Type of scipy interpolation to use for depth interpolation
                                [default = 'linear']
         nemo_frequency (str) : Time frequency of NEMO daily. String inputs can
                                be 'hourly', 'daily' or 'monthly'. Alternatively,
                                provide an integer number of hours.
         en4_sal_name (str)   : Name of EN4 (COAST) variable to use for salinity
                                [default = 'practical_salinity']
         en4_tem_name (str)   : Name of EN4 (COAST) variable to use for temperature
                                [default = 'potential_temperature']
         fn_mesh_mask (str)   : Full path to mesh_mask.nc if data is NEMO v3.6
         start_date (datetime): Start date for EN4 data. If not provided, script
                                will make a guess based on nemo_frequency.
         end_date (datetime)  : End date for EN4 data. If not provided, script
                                will make a guess based on nemo_frequency.
                                
        OUTPUTS:
         Returns a new PROFILE object containing an uncomputed dataset and 
         can write to file (recommended):
         Writes extracted data to file. Extracted dataset has the dimensions:
        '''
            
        # Read EN4, then extract desired variables
        en4 = self.dataset
        nemo = nemo.dataset
        
        # CHECKS
        # 1. Check that bottom_level is in dataset if mask_bottom_level is True
        if mask_bottom_level:
            if 'bottom_level' not in nemo.variables:
                raise ValueError('bottom_level not found in input dataset. Please ensure variable is present or set mask_bottom_level to False')
        
        # Use only observations that are within model time window.
        en4_time = en4.time.values
        mod_time = nemo.time.values
        
        # SPATIAL indices - nearest neighbour
        ind2D = general_utils.nearest_indices_2D(nemo['longitude'], nemo['latitude'],
                                           en4['longitude'], en4['latitude'], 
                                           mask=nemo.landmask)
        ind_x = ind2D[0]
        ind_y = ind2D[1]
        print('Spatial Indices Calculated', flush=True)
        
        # TIME indices - model nearest to obs time
        en4_time = en4.time.values
        ind_t = [ np.argmin( np.abs( mod_time - en4_time[tt] ) ) for tt in range(en4.dims['profile'])]
        ind_t = xr.DataArray(ind_t)
        print('Time Indices Calculated', flush=True)
        
        # Find out which variables have both depth and profile
        # This is for applying the bottom_level mask later
        var_list = list(nemo.keys())
        bl_var_list = []
        for vv in var_list:
            cond1 = 'z_dim' not in nemo[vv].dims
            if cond1:
                bl_var_list.append(vv)
        
        # Get chunks along the time dimension and determine whether chunks
        # are described by a single equal size, or a tuples of sizes
        time_chunks = nemo.chunks['t_dim']
        time_dim = nemo.dims['t_dim']
        start_ii = 0 # Starting index for loading data. Increments each loop
        count_ii = 0 # Counting index for allocating data. Increments 1 each loop
        
        while start_ii < time_dim:
            end_ii = start_ii+time_chunks[count_ii]
            #print('{0}: {1} > {2}'.format(count_ii, start_ii, end_ii))
            
            # Determine which time indices lie in this chunk
            ind_in_chunk = np.logical_and(ind_t>=start_ii, ind_t<end_ii)
            
            # Check There are some indices at all
            if np.sum(ind_in_chunk) == 0:
                start_ii = end_ii
                count_ii = count_ii + 1
                continue
            
            # Pull out x,y and t indices
            ind_x_in_chunk = ind_x[ind_in_chunk]
            ind_y_in_chunk = ind_y[ind_in_chunk]
            ind_t_in_chunk = ind_t[ind_in_chunk] - start_ii
            
            # Index a temporary chunk and read it to memory
            ds_tmp = nemo.isel(t_dim=np.arange(start_ii, end_ii)).load()
            
            # Index loaded chunk and rename dim_0 to profile
            ds_tmp_indexed = ds_tmp.isel(x_dim = ind_x_in_chunk, 
                                         y_dim = ind_y_in_chunk,
                                         t_dim = ind_t_in_chunk)
            ds_tmp_indexed = ds_tmp_indexed.rename({'dim_0':'profile'})
            
            # Mask out all levels deeper than bottom_level
            # Here I have used set_coords() and reset_coords() to omit variables
            # with no z_dim from the masking. Otherwise xr.where expands these 
            # dimensions into full 2D arrays.
            if mask_bottom_level:
                n_z_tmp = ds_tmp_indexed.dims['z_dim']
                bl_array = ds_tmp_indexed.bottom_level.values
                z_index, bl_index = np.meshgrid(np.arange(0,n_z_tmp), 
                                                bl_array)
                mask2 = xr.DataArray(z_index < bl_index, dims=['profile','z_dim'])
                ds_tmp_indexed = ds_tmp_indexed.set_coords(bl_var_list)
                ds_tmp_indexed = ds_tmp_indexed.where(mask2)
                ds_tmp_indexed = ds_tmp_indexed.reset_coords(bl_var_list)
            
            # If not first iteration, concatenate this indexed chunk onto
            # final output dataset
            if count_ii == 0:
                mod_profiles = ds_tmp_indexed
            else:
                mod_profiles = xr.concat((mod_profiles, ds_tmp_indexed),
                                           dim = 'profile')
                
            # Update counters
            start_ii = end_ii
            count_ii = count_ii + 1
            
        
        # Put obs time into the output array
        mod_profiles['obs_time'] = (['profile'], en4_time)
        
        # Calculate interpolation distances
        interp_dist = general_utils.calculate_haversine_distance(
                                             en4.longitude, 
                                             en4.latitude, 
                                             mod_profiles.longitude, 
                                             mod_profiles.latitude)
        mod_profiles['interp_dist'] = (['profile'], interp_dist.values)
        
        # Calculate interpolation time lags
        interp_lag = (mod_profiles.time.values - en4_time).astype('timedelta64[h]')
        mod_profiles['interp_lag'] = (['profile'], interp_lag)
            
        # Create return object and put dataset into it.
        return_prof = Profile()
        return_prof.dataset = mod_profiles
        return return_prof