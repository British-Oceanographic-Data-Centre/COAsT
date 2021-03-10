import numpy as np
import matplotlib.pyplot as plt
from . import stats_util
from .logging_util import get_slug, debug, error
import scipy.stats
import xarray as xr

class DISTRIBUTION:
    '''
    An object for storing Cumulative Distribution Function information.
    Used primarily for calculating the Continuous Ranked Probability Score.
    The object is initalisated by passing in a 'sample' vector of data for
    which to determine a CDF. This sample is used to construct either a 
    empirical or theoretical CDF, depending on the cdf_type argument.
    '''
    
    def __init__(self, sample):
        """Initialisation of CDF object.
        Args:
            sample (array): Data sample over which to estimate CDF
            cdf_type (str): Either 'empirical' or 'theoretical'.
            cdf_func (str): Function type if cdf_type='theoretical'. Presently
                            only 'gaussian'.
        Returns:
            New CDF object.
        """
        debug(f"Creating a new {get_slug(self)}")
        self.sample = sample
        self.sample_size = len(sample)
        self.mu       = np.nanmean(sample)
        self.sigma    = np.nanstd(sample)
        self.plot_xmin, self.plot_xmax = self.set_x_bounds()
        debug(f"{get_slug(self)} initialised")
        
    def set_x_bounds(self):
        ''' Calculate x bounds for CDF plotting '''
        # Is input a single value (st. dev == 0)
        single_value = True if self.sigma == 0 else False  # TODO This could just be: single_value = self.sigma == 0
        if single_value: 
            self.cdf_type = 'empirical'
        
        # Calculate x bounds as 5 std dev. either side of the mean
        debug(f"Calculating x bounds for {get_slug(self)}")
        if single_value:
            xmin = self.mu-1
            xmax = self.mu+1
        else:
            xmin = self.mu - 5*self.sigma
            xmax = self.mu + 5*self.sigma
        return xmin, xmax
    
    def build_discrete_cdf(self, x: np.ndarray = None, n_pts: int = 1000):
        """Builds a discrete CDF for plotting and direct comparison.
        Args:
            x (array): x-values over which to calculate discrete CDF values.
                       If none, these will be determined using mu and sigma.
            n_pts (int): n_pts to use for x if x=None.
        Returns:
            x and y arrays for discrete CDF.
        """
        debug(f"Discrete CDF will be built for {get_slug(self)}")
        # Build discrete X bounds
        if x is None:
            debug(f"Building discrete X bounds for {get_slug(self)}")
            x = np.linspace(self.plot_xmin, self.plot_xmax, n_pts)
        
        if self.cdf_type == "empirical":
            y = stats_util.empirical_distribution(x, self.sample)

        else:
            error(f"CDF type for {get_slug(self)} is , which is not acceptable, raising exception!")

        return x, y
        
    @staticmethod
    def normal_distribution(mu: float=0, sigma: float=1, 
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
        debug(f"Generating normal distribution for {get_slug(x)}")
        term1 = sigma*np.sqrt( 2*np.pi )
        term1 = 1/term1
        exponent = -0.5*((x-mu)/sigma)**2
        return term1*np.exp( exponent )
    
    @staticmethod
    def cumulative_distribution(mu: float=0, sigma: float=1, 
                                x: np.ndarray=None, cdf_func: str='gaussian'):
        """Integrates under a discrete PDF to obtain an estimated CDF.
    
        Keyword arguments:
        x   -- Arbitrary array of x-values
        pdf -- PDF corresponding to values in x. E.g. as generated using
               normal_distribution.
        
        return: Array of len(x) containing the discrete cumulative values 
                estimated using the integral under the provided PDF.
        """
        debug(f"Estimating CDF using {get_slug(x)}")
        if cdf_func=='gaussian': #If Gaussian, integrate under pdf
            pdf = DISTRIBUTION.normal_distribution(mu=mu, sigma=sigma, x=x)
            cdf = [np.trapz(pdf[:ii],x[:ii]) for ii in range(0,len(x))]
        else: 
            raise NotImplementedError
        return np.array(cdf)
    
    @staticmethod
    def empirical_distribution(x, sample):
        """Estimates a CDF empirically.
    
        Keyword arguments:
        x      -- Array of x-values over which to generate distribution
        sample -- Sample to use to generate distribution
        
        return: Array of len(x) containing corresponding EDF values
        """
        debug(f"Estimating empirical distribution with {get_slug(x)}")
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        edf = np.zeros(len(x))
        n_sample = len(sample)
        for ss in sample:
            edf[x>ss] = edf[x>ss] + 1/n_sample
        return xr.DataArray(edf)
        
    def get_common_x(self, other, n_pts=2000):
        """Generates a common x vector for two CDF objects."""
        debug(f"Generating common X vector for {get_slug(self)} and {get_slug(other)}")
        xmin = min(self.plot_xmin, other.plot_xmin)
        xmax = max(self.plot_xmax, other.plot_xmax)
        common_x = np.linspace(xmin, xmax, n_pts)
        return common_x
    
    def plot_cdf(self):
        """ A quick plot showing the CDF contained in this object."""
        debug(f"Generating quick plot for {get_slug(self)}")
        ax = plt.subplot(111)
        x,y = self.build_discrete_cdf()
        ax.plot(x, y)
        ax.grid()
        return
    
    def integrate_cdf(self, other=None, plot=False):
        """Returns the integral under CDF or between two CDFs. This is 
        equivalent to the first order Wasserstein metric for two 
        probability distributions"""
        debug(f"Generating diff plot for {get_slug(self)} and {get_slug(other)}")

        if other is None:
            integral = scipy.stats.wasserstein_distance(self.sample, [0])
            if plot:
                x = self.get_common_x(other)
                x, y1 = self.build_discrete_cdf(x)
                dum, y2 = other.build_discrete_cdf(x)
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x, y1, c='k', linestyle='--')
                ax.plot(x, y2, linestyle='--')
                ax.fill_between(x, y1, y2, alpha=0.5)
                ax.set_title('Area: ' + str(integral))
                plt.legend(('1','2'))
            
        else:
            integral = scipy.stats.wasserstein_distance(self.sample, other.sample)
            
            if plot:
                x = self.get_common_x(other)
                x, y1 = self.build_discrete_cdf(x)
                dum, y2 = other.build_discrete_cdf(x)
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x, y1, c='k', linestyle='--')
                ax.plot(x, y2, linestyle='--')
                ax.fill_between(x, y1, y2, alpha=0.5)
                ax.set_title('Area: ' + str(integral))
                plt.legend(('1','2'))
                
        if plot:
            return integral, fig, ax
        else:
            return integral