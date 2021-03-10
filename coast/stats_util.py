'''
Python definitions used to aid with statistical calculations.

*Methods Overview*
    -> normal_distribution(): Create values for a normal distribution
    -> cumulative_distribution(): Integration udner a PDF
    -> empirical_distribution(): Estimates CDF empirically
'''

import numpy as np
import xarray as xr
from .logging_util import get_slug, debug, info, warn, error
import scipy

def find_maxima(x, y, method='comp', **kwargs):
    '''
    Finds maxima of a time series y. Returns maximum values of y (e.g heights)
    and corresponding values of x (e.g. times). 
    **kwargs are dependent on method.
    
        Methods:
        'comp' :: Find maxima by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        DB NOTE: Currently only the 'comp' method is implemented. Future
                 methods include linear interpolation and cublic splines.
    '''
    if method == 'comp':
        peaks, props = scipy.signal.find_peaks(y, **kwargs)
        return x[peaks], y[peaks]

def doodson_x0_filter(elevation, ax=0):
    ''' 
    The Doodson X0 filter is a simple filter designed to damp out the main 
    tidal frequencies. It takes hourly values, 19 values either side of the 
    central one and applies a weighted average using:
              (1010010110201102112 0 2112011020110100101)/30.
    ( http://www.ntslf.org/files/acclaimdata/gloup/doodson_X0.html )
    
    In "Data Analaysis and Methods in Oceanography":
    
    "The cosine-Lanczos filter, the transform filter, and the
    Butterworth filter are often preferred to the Godin filter,
    to earlier Doodson filter, because of their superior ability
    to remove tidal period variability from oceanic signals."
    
    This routine can be used for any dimension input array.
    
    Parameters
    ----------
        elevation (ndarray) : Array of hourly elevation values.
        axis (int) : Time axis of input array. This axis must have >= 39 
        elements
       
    Returns
    -------
        Filtered array of same rank as elevation.
    ''' 
    if elevation.shape[ax] < 39:
        print('Doodson_XO: Ensure time axis has >=39 elements. Returning.')
        return
    # Define DOODSON XO weights
    kern = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 1, 1, 2, 
                     0,
                     2, 1, 1, 2, 0, 1, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    kern = kern/30

    # Convolve input array with weights along the specified axis.
    filtered = np.apply_along_axis(lambda m: np.convolve(m, kern, mode=1), 
                                   axis=ax, arr=elevation)

    # Pad out boundary areas with NaNs for given (arbitrary) axis.
    # DB: Is this the best way to do this?? Can put_along_axis be used instead
    filtered = filtered.swapaxes(0,ax)
    filtered[:19] = np.nan
    filtered[-19:] = np.nan
    filtered = filtered.swapaxes(0,ax)
    return filtered