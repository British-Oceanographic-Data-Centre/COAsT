import xarray as xr
import numpy as np
from .COAsT import COAsT
import matplotlib.pyplot as plt

class CDF():
    '''
    An object for storing Cumulative Distribution Function information.
    '''
    
    def __init__(self, sample, cdf_type='empirical', cdf_func = 'gaussian'):
        self.cdf_type = cdf_type
        self.cdf_func = cdf_func
        self.sample   = sample
        self.sample_size = sample.shape
        self.mu       = np.nanmean(sample)
        self.sigma    = np.nanstd(sample)
        self.disc_x   = None # Discrete CDF x-values
        self.disc_y   = None # Discrete CDF y-values
        self._cdf_type_options = ['empirical', 'theoretical']
        self._cdf_func_options = ['gaussian']
        self.build_discrete_cdf()
    
    def build_discrete_cdf(self, x: np.ndarray=None, n_pts: int=1000):
        '''
        
        '''
        # Is input a single value (st. dev == 0)
        single_value = True if self.sigma == 0 else False
        
        # Build discrete X values according to CDF function if x is unspecified
        if x is None:
            if single_value:
                x = np.linspace(self.mu-1, self.mu+1, n_pts)
            elif self.cdf_func == 'gaussian':
                x = np.linspace( self.mu-5*self.sigma, self.mu+5*self.sigma, 
                                n_pts)
            
        # Build discrete Y values according to CDF type, CDF function 
        # (if theoretical) and sample.
        if single_value: self.cdf_type = 'empirical'
        
        if self.cdf_type == "empirical":
            y = self.empirical_distribution(x, self.sample)
            
        elif self.cdf_type == "theoretical": 
            if self.cdf_func == 'gaussian':
                y = self.cumulative_distribution(mu=self.mu, sigma=self.sigma,
                                                 x=x, cdf_func=self.cdf_func)
            else:
                raise NotImplementedError
        else:
            raise Exception('CDF Type must be empirical or theoretical')

        self.disc_x = x
        self.disc_y = y
        return
    
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
    
    def empirical_distribution(self,x, sample):
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
    
    def difference(self, other):
        """Calculated the CRPS of provided model and observed CDFs.

        Keyword arguments:
        cdf1 -- Discrete CDF of model data
        cdf2   -- Discrete CDF of observation data
        
        return: A single squared difference between two CDFs.
        """
        xmin = min(self.disc_x[0], other.disc_x[0])
        xmax = max(self.disc_x[-1], other.disc_x[-1])
        common_x = np.linspace(xmin, xmax, 1000)
        self.build_discrete_cdf(x=common_x)
        other.build_discrete_cdf(x=common_x)
        return np.trapz((other.disc_y - self.disc_x)**2, common_x)
    
    def quick_plot(self):
        ax = plt.subplot(111)
        ax.plot(self.disc_x, self.disc_y)
        ax.grid()
        return