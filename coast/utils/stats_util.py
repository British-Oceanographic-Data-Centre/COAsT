import numpy as np
import xarray as xa
from warnings import warn
import matplotlib.pyplot as plt

def normal_distribution(self, mu: float=0, sigma: float=1, 
                        x: np.ndarray=None, n_pts: int=1000):
    """Generates a discrete normal distribution.

    Keyword arguments:
    x     -- Arbitrary array of x-values
    mu    -- Distribution mean
    sigma -- Distribution standard deviation
    
    return: Array of len(x) containing the normal values calculated from
            the elements of x.
    """
    if x is None:
        x = np.linspace( mu-5*sigma, mu+5*sigma, n_pts)
    term1 = sigma*np.sqrt( 2*np.pi )
    term1 = 1/term1
    exponent = -0.5*((x-mu)/sigma)**2
    return term1*np.exp( exponent )

def cumulative_distribution(self, mu: float=0, sigma: float=1, 
                            x: np.ndarray=None, cdf_func: str='gaussian'):
    """Integrates under a discrete PDF to obtain an estimated CDF.

    Keyword arguments:
    x   -- Arbitrary array of x-values
    pdf -- PDF corresponding to values in x. E.g. as generated using
           normal_distribution.
    
    return: Array of len(x) containing the discrete cumulative values 
            estimated using the integral under the provided PDF.
    """
    if cdf_func=='gaussian': #If Gaussian, integrate under pdf
        pdf = self.normal_distribution(mu=mu, sigma=sigma, x=x)
        cdf = [np.trapz(pdf[:ii],x[:ii]) for ii in range(0,len(x))]
    else: 
        raise NotImplementedError
    return np.array(cdf)

def empirical_distribution(self, x, sample):
    """Estimates a CDF empirically.

    Keyword arguments:
    x      -- Array of x-values over which to generate distribution
    sample -- Sample to use to generate distribution
    
    return: Array of len(x) containing corresponding EDF values
    """
    sample = np.array(sample)
    sample = sample[~np.isnan(sample)]
    sample = np.sort(sample)
    edf = np.zeros(len(x))
    n_sample = len(sample)
    for ss in sample:
        edf[x>ss] = edf[x>ss] + 1/n_sample
    return xr.DataArray(edf)