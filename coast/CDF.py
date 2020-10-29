import numpy as np
import matplotlib.pyplot as plt
from . import stats_util
from .logging_util import get_slug, debug, error


class CDF:
    '''
    An object for storing Cumulative Distribution Function information.
    Used primarily for calculating the Continuous Ranked Probability Score.
    The object is initalisated by passing in a 'sample' vector of data for
    which to determine a CDF. This sample is used to construct either a 
    empirical or theoretical CDF, depending on the cdf_type argument.
    '''
    
    def __init__(self, sample, cdf_type: str='empirical', 
                 cdf_func: str='gaussian'):
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
        self.cdf_type = cdf_type
        self.cdf_func = cdf_func
        self.sample = sample
        self.sample_size = len(sample)
        self.mu       = np.nanmean(sample)
        self.sigma    = np.nanstd(sample)
        self._cdf_type_options = ['empirical', 'theoretical']
        self._cdf_func_options = ['gaussian']
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
        elif self.cdf_func == 'gaussian':
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

        elif self.cdf_type == "theoretical":
            if self.cdf_func == 'gaussian':
                y = stats_util.cumulative_distribution(
                                                 mu=self.mu, sigma=self.sigma,
                                                 x=x, cdf_func=self.cdf_func)
            else:
                raise NotImplementedError
        else:
            error(f"CDF type for {get_slug(self)} is \"{self.cdf_type}\", which is not acceptable, raising exception!")
            raise Exception(f'CDF Type must be empirical or theoretical')  # TODO This should probably be a ValueError
        debug(f"CDF type for {get_slug(self)} is \"{self.cdf_type}\"")

        return x, y
    
    def get_common_x(self, other, n_pts=5000):
        """Generates a common x vector for two CDF objects."""
        debug(f"Generating common X vector for {get_slug(self)} and {get_slug(other)}")
        xmin = min(self.plot_xmin, other.plot_xmin)
        xmax = max(self.plot_xmax, other.plot_xmax)
        common_x = np.linspace(xmin, xmax, n_pts)
        return common_x
    
    def quick_plot(self):
        """ A quick plot showing the CDF contained in this object."""
        debug(f"Generating quick plot for {get_slug(self)}")
        ax = plt.subplot(111)
        x,y = self.build_discrete_cdf()
        ax.plot(x, y)
        ax.grid()
        return
    
    def integral(self, other, plot=False):
        """Plots two CDFS on one plot, with the difference shaded"""
        debug(f"Generating diff plot for {get_slug(self)} and {get_slug(other)}")
    
        x = self.get_common_x(other)
        x, y1 = self.build_discrete_cdf(x)
        dum, y2 = other.build_discrete_cdf(x)
        integral = np.abs(np.trapz(x,y1) - np.trapz(x,y2))
        if plot:
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
