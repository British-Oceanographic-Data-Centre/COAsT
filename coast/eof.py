import xarray as xr
import numpy as np
from scipy import linalg
from scipy import signal

def eof(variable:xr.DataArray, full_matrices=False, time_dim_name:str='t_dim'):
    variable = variable.transpose(...,time_dim_name)
    variable_mask = np.ma.masked_invalid(variable).mask
    I,J,T = np.shape(variable.data)
    F = np.reshape(variable.data,(I*J, T))
                  
    # Remove constant zero point such as land points
    active_ind = np.where( np.nan_to_num(F).any(axis=1))[0] 
    F = F[active_ind,:] 
    # Remove time mean at each grid point
    mean = F.mean(axis=1)
    F = F - mean[:, np.newaxis]
    
    # Calculate eofs and pcs using SVD        
    P, D, Q = linalg.svd( F, full_matrices=full_matrices )
    EOFs = np.zeros( (I*J,T), dtype=P.dtype)
    EOFs[active_ind,:] = P 
    
    # Calculate variance explained
    if full_matrices:        
        variance_explained = 100.*( D**2 / np.sum( D**2 ) )
    else:
        Inv = P.dot( np.dot( np.diag(D), Q ) ) # PDQ
        var1 = np.sum( np.var( Inv, axis=1, ddof=1 ) )
        var2 = np.sum( np.var( F, axis=1, ddof=1 ) )
        mult = var1 / var2
        variance_explained = 100.*mult*( D**2 / np.sum( D**2 ) )
    
    # Reshape and scale PCs
    PCs = np.transpose(Q) * D
    scale = np.max(np.abs(PCs),axis=0)
    EOFs = np.reshape(EOFs, (I,J,T)) 
    PCs = np.transpose(Q) * D 
         
    # Assign to xarray variables
    # copy the coordinates 
    coords = {'mode':(('mode'),np.arange(1,T+1))}
    time_coords = {'mode':(('mode'),np.arange(1,T+1))}
    for coord in variable.coords:
        if variable.dims[2] not in variable[coord].dims:
            coords[coord] = (variable[coord].dims, variable[coord])
        else:
            if (variable.dims[2],) == variable[coord].dims:
                time_coords[coord] = (variable[coord].dims, variable[coord])
    dataset = xr.Dataset()
    dims = (variable.dims[:2]) + ('mode',)
    dataset['EOF'] = (xr.DataArray(np.ma.masked_array( 
                        EOFs, mask=variable_mask), coords=coords, dims=dims))
    dataset.EOF.attrs['standard name'] = 'EOFs'
    
    dims = (variable.dims[2],'mode')
    dataset['PC'] = xr.DataArray(PCs, coords=time_coords, dims=dims)
    
    dataset['variance'] = (xr.DataArray(variance_explained, 
            coords={'mode':(('mode'),np.arange(1,T+1))}, dims=['mode']))
    
    return dataset

    
def hilbert_eof(variable:xr.DataArray, full_matrices=False, time_dim_name:str='t_dim'):
    variable = variable.transpose(...,time_dim_name)
    variable_mask = np.ma.masked_invalid(variable).mask
    I,J,T = np.shape(variable.data)
    F = np.reshape(variable.data,(I*J, T))
                  
    # Remove constant zero point such as land points
    active_ind = np.where( np.nan_to_num(F).any(axis=1))[0] 
    F = F[active_ind,:]
    # Remove time mean at each grid point
    mean = np.mean(F,axis=1)
    F = F - mean[:, np.newaxis]
    # Apply Hilbert transform
    F = signal.hilbert(F, axis=1)
    
    # Calculate eofs and pcs using SVD 
    P, D, Q = linalg.svd(F, full_matrices=full_matrices)        
    EOFs = np.zeros( (I*J, T), dtype=P.dtype)
    EOFs[active_ind,:] = P 
    
    # Calculate variance explained
    if full_matrices:        
        variance_explained = 100.*( D**2 / np.sum( D**2 ) )
    else:
        Inv = P.dot( np.dot( np.diag(D), Q ) ) # PDQ
        var1 = np.sum( np.var( Inv, axis=1, ddof=1 ) )
        var2 = np.sum( np.var( F, axis=1, ddof=1 ) )
        mult = var1 / var2
        variance_explained = 100.*mult*( D**2 / np.sum( D**2 ) )
    
    # Extract amplitude and phase of the time component
    PCs = np.transpose(Q) * D
    PC_amp = np.absolute(PCs)
    PC_phase = np.angle(PCs)
    
    # Extract the amplitude and phase of the spatial component
    EOFs = np.reshape(EOFs, (I,J,T))             
    EOF_amp =  np.absolute(EOFs)
    EOF_phase = np.angle(EOFs)

    # Assign to xarray variables
    # copy the coordinates 
    dataset = xr.Dataset()
    coords = {'mode':(('mode'),np.arange(1,T+1))}
    time_coords = {'mode':(('mode'),np.arange(1,T+1))}
    for coord in variable.coords:
        if variable.dims[2] not in variable[coord].dims:
            coords[coord] = (variable[coord].dims, variable[coord])
        else:
            if (variable.dims[2],) == variable[coord].dims:
                time_coords[coord] = (variable[coord].dims, variable[coord])
                
    dims = (variable.dims[:2]) + ('mode',)
    dataset['EOF_amp'] = (xr.DataArray(np.ma.masked_array( 
                    EOF_amp, mask=variable_mask), coords=coords, dims=dims))
    dataset.EOF_amp.attrs['standard_name'] = 'EOF amplitude'
    dataset['EOF_phase'] = (xr.DataArray( np.ma.masked_array( 
        np.rad2deg(EOF_phase), mask=variable_mask), coords=coords, dims=dims))
    dataset.EOF_phase.attrs['standard_name'] = 'EOF phase'
    dataset.EOF_phase.attrs['units'] = 'degrees'
 
    dims = (variable.dims[2],'mode')
    dataset['temporal_amp'] = xr.DataArray(PC_amp, coords=time_coords, dims=dims)
    dataset.temporal_amp.attrs['standard_name'] = 'temporal projection amplitude'
    dataset['temporal_phase'] = (xr.DataArray( 
        np.rad2deg(PC_phase), coords=time_coords, dims=dims))
    dataset.temporal_amp.attrs['standard_name'] = 'temporal projection amplitude'

    dataset['variance'] = (xr.DataArray(variance_explained, 
            coords={'mode':(('mode'),np.arange(1,T+1))}, dims=['mode']))
    
    return dataset
    
   
                
        
        