import xarray as xr
import numpy as np
from scipy import linalg
from scipy.signal import hilbert


def eofs(variable:xr.DataArray, full_matrices:bool=False, time_dim_name:str='t_dim'):
    '''
    Computes the Empirical Orthogonal Functions (EOFs) of a variable (time series) 
    that has 3 dimensions where one is time, i.e. (x,y,time)
    
    Returns the set of EOF modes, the associated temporal projections and the 
    variance explained by each mode as DataArrays within an xarray Dataset.
    
    All-NaN time series, such as those at land points, are handled and ignored;
    however, isolated NaNs within a time series, i.e. missing data point, must 
    be filled before calling the function. 
    
    The variable will be de-meaned in time before the EOFs are computed, normalisation
    should be carried out before calling the function if desired. The returned EOFs and 
    temporal projections are not scaled or normalised.

    Parameters
    ----------
    variable : (xarray.DataArray), 3-dimensional variable of size (I,J,T), 
    containing I*J time series
    full_matrices : (boolean, default False) if false computes only first K EOFs 
    where K=min(I*J,T), where T is total number of time points. Setting to True
    could demand significant memory.
    time_dim_name : (string, default 't_dim') the name of the time dimension.

    Returns
    -------
    dataset : xarray Dataset, containing the EOFs, temporal projections and 
    variance explained as xarray DataArrays. The relevent coordinates 
    from the original data variable are also included   

    '''
    
    variable = variable.transpose(...,time_dim_name,transpose_coords=False)
    I,J,T = np.shape(variable.data)
    signal = np.reshape(variable.data,(I*J, T))
                  
    # Remove constant zero point such as land points
    active_ind = np.where( np.nan_to_num(signal).any(axis=1))[0] 
    signal = signal[active_ind,:] 
    # Remove time mean at each grid point
    mean = signal.mean(axis=1)
    signal = signal - mean[:, np.newaxis]

    EOFs, projections, variance_explained, mode_count = \
                            _compute( signal, full_matrices, active_ind, I*J )

    EOFs = np.reshape(EOFs, (I,J,mode_count)) 
    
    # Assign to xarray variables
    # copy the coordinates 
    coords = {'mode':(('mode'),np.arange(1,mode_count+1))}
    time_coords = {'mode':(('mode'),np.arange(1,mode_count+1))}
    for coord in variable.coords:
        if variable.dims[2] not in variable[coord].dims:
            coords[coord] = (variable[coord].dims, variable[coord])
        else:
            if (variable.dims[2],) == variable[coord].dims:
                time_coords[coord] = (variable[coord].dims, variable[coord])
    
    dataset = xr.Dataset()
    
    dims = (variable.dims[:2]) + ('mode',)
    dataset['EOF'] = xr.DataArray( EOFs, coords=coords, dims=dims)
    dataset.EOF.attrs['standard name'] = 'EOF'
    
    dims = (variable.dims[2],'mode')
    dataset['temporal_proj'] = xr.DataArray(projections, coords=time_coords, dims=dims)
    dataset.temporal_proj.attrs['standard name'] = 'temporal projection'
    
    dataset['variance'] = (xr.DataArray(variance_explained, 
            coords={'mode':(('mode'),np.arange(1,mode_count+1))}, dims=['mode']))
    dataset.variance.attrs['standard name'] = 'percentage of variance explained'
      
    return dataset

    
def hilbert_eofs(variable:xr.DataArray, full_matrices=False, time_dim_name:str='t_dim'):
    '''
    Computes the complex Hilbert Empirical Orthogonal Functions (HEOFs) of a 
    variable (time series) that has 3 dimensions where one is time, i.e. (x,y,time).
    See https://doi.org/10.1002/joc.1499
    
    Returns the set of HEOF amplitude and phase modes, the associated temporal 
    projection amplitudes and phases and the variance explained by each mode 
    as DataArrays within an xarray Dataset.
    
    All-NaN time series, such as those at land points, are handled and ignored;
    however, isolated NaNs within a time series, i.e. missing data point, must 
    be filled before calling the function. 
    
    The variable will be de-meaned in time before the EOFs are computed, normalisation
    should be carried out before calling the function if desired. The returned EOFs and 
    temporal projections are not scaled or normalised.

    Parameters
    ----------
    variable : (xarray.DataArray), 3-dimensional variable of size (I,J,T), 
    containing I*J time series
    full_matrices : (boolean, default False) if false computes only first K EOFs 
    where K=min(I*J,T), where T is total number of time points.
    time_dim_name : (string, default 't_dim') the name of the time dimension.

    Returns
    -------
    dataset : xarray Dataset, containing the EOF amplitudes and phases, 
    temporal projection amplitude and phases and the variance explained 
    as xarray DataArrays. The relevent coordinates 
    from the original data variable are also in the dataset.       

    '''
    variable = variable.transpose(...,time_dim_name,transpose_coords=False)
    I,J,T = np.shape(variable.data)
    signal = np.reshape(variable.data,(I*J, T))
                  
    # Remove constant zero point such as land points
    active_ind = np.where( np.nan_to_num(signal).any(axis=1))[0] 
    signal = signal[active_ind,:]
    # Remove time mean at each grid point
    mean = np.mean(signal,axis=1)
    signal = signal - mean[:, np.newaxis]
    # Apply Hilbert transform
    signal = hilbert(signal, axis=1)
    # Compute EOFs
    EOFs, projections, variance_explained, mode_count = \
                                _compute( signal, full_matrices, active_ind, I*J )
    
    # Extract the amplitude and phase of the projections    
    projection_amp = np.absolute(projections)
    projection_phase = np.angle(projections)
    
    # Extract the amplitude and phase of the spatial component
    EOFs = np.reshape(EOFs, (I,J,mode_count))             
    EOF_amp =  np.absolute(EOFs)
    EOF_phase = np.angle(EOFs)

    # Assign to xarray variables
    # copy the coordinates 
    dataset = xr.Dataset()
    coords = {'mode':(('mode'),np.arange(1,mode_count+1))}
    time_coords = {'mode':(('mode'),np.arange(1,mode_count+1))}
    for coord in variable.coords:
        if variable.dims[2] not in variable[coord].dims:
            coords[coord] = (variable[coord].dims, variable[coord])
        else:
            if (variable.dims[2],) == variable[coord].dims:
                time_coords[coord] = (variable[coord].dims, variable[coord])
                
    dims = (variable.dims[:2]) + ('mode',)
    dataset['EOF_amp'] = xr.DataArray(EOF_amp, coords=coords, dims=dims)
    dataset.EOF_amp.attrs['standard_name'] = 'EOF amplitude'
    dataset['EOF_phase'] = xr.DataArray(np.rad2deg(EOF_phase), coords=coords, dims=dims)
    dataset.EOF_phase.attrs['standard_name'] = 'EOF phase'
    dataset.EOF_phase.attrs['units'] = 'degrees'
 
    dims = (variable.dims[2],'mode')
    dataset['temporal_amp'] = xr.DataArray(projection_amp, coords=time_coords, dims=dims)
    dataset.temporal_amp.attrs['standard_name'] = 'temporal projection amplitude'
    dataset['temporal_phase'] = (xr.DataArray( 
        np.rad2deg(projection_phase), coords=time_coords, dims=dims))
    dataset.temporal_phase.attrs['standard_name'] = 'temporal projection phase'
    dataset.temporal_phase.attrs['units'] = 'degrees'

    dataset['variance'] = (xr.DataArray(variance_explained, 
            coords={'mode':(('mode'),np.arange(1,mode_count+1))}, dims=['mode']))
    dataset.variance.attrs['standard name'] = 'percentage of variance explained'
    
    return dataset


def _compute(signal, full_matrices, active_ind, number_points):
    '''
    Compute eofs, projections and variance explained using a Singular Value Decomposition

    Parameters
    ----------
    signal : (array) the signal
    full_matrices : (boolean) whether to return a full or abbreviated SVD
    active_ind : (array) indices of points with non-null signal
    number_points : (int) number of points in original data set

    Returns
    -------
    EOFs : (array) the EOFs in 2d form 
    projections : (array) the projectsion of the EOFs
    variance_explained : (array) variance explained by each mode
    mode_count : (int) number of modes computed

    '''
    P, D, Q = linalg.svd(signal, full_matrices=full_matrices)   
    mode_count = P.shape[-1]     
    EOFs = np.zeros( (number_points, mode_count), dtype=P.dtype)
    EOFs[active_ind,:] = P 
    
    # Calculate variance explained
    if full_matrices:        
        variance_explained = 100.*( D**2 / np.sum( D**2 ) )
    else:
        Inv = P.dot( np.dot( np.diag(D), Q ) ) # PDQ
        var1 = np.sum( np.var( Inv, axis=1, ddof=1 ) )
        var2 = np.sum( np.var( signal, axis=1, ddof=1 ) )
        mult = var1 / var2
        variance_explained = 100.*mult*( D**2 / np.sum( D**2 ) )
    
    # Extract EOF projections
    projections = np.transpose(Q) * D
      
    return EOFs, projections, variance_explained, mode_count

   
                
        
        