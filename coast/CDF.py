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
    
    def crps(self, xa):
        """Calculate CRPS as outlined in Hersbach et al. 2000

        Args:
            xa (float): A single 'observation' value which to compare against
                        CDF.

        Returns:
            A single CRPS value.
        """

        debug(f"Calculating {get_slug(self)} CRPS, compare against {xa}")
        
        def calc(alpha, beta, p):
            return alpha * p**2 + beta*(1 - p)**2
        crps_sum = 0
        sample = np.array(self.sample).flatten()
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        sample_size = len(sample)
        
        # All intervals within range of the sample distribution
        for ii in range(0, sample_size-1):
            p = (ii+1)/sample_size
            if xa > sample[ii+1]:
                alpha = sample[ii+1] - sample[ii]
                beta = 0
            elif xa < sample[ii]:
                alpha = 0
                beta = sample[ii+1] - sample[ii]
            else:
                alpha = xa - sample[ii]
                beta = sample[ii+1] - xa
            crps = calc(alpha, beta, p)
            crps_sum = crps_sum + crps
        # Intervals 0 and N, where p = 0 and 1 respectively
        if xa < sample[0]:
            p=0
            alpha = 0
            beta = sample[0] - xa
            crps = calc(alpha, beta, p)
            crps_sum = crps_sum + crps
        elif xa > sample[-1]:
            p=1
            alpha = xa - sample[-1]
            beta = 0
            crps = calc(alpha, beta, p)
            crps_sum = crps_sum + crps
            
        return crps_sum
    
    def crps_fast(self, xa):
        """Faster version of crps method, using multiple numpy arrays.
           For large sample sizes, this is considerably faster but uses more
           memory.

        Args:
            xa (float): A single 'observation' value which to compare against
                        CDF.

        Returns:
            A single CRPS value.
        """
        debug(f"Calculating {get_slug(self)} fast CRPS, compare against {xa}")
        def calc(alpha, beta, p):
            return alpha * p**2 + beta*(1 - p)**2
        xa = float(xa)
        crps_sum = 0
        sample = np.array(self.sample).flatten()
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        sample_size = len(sample)
        
        alpha = np.zeros(sample_size-1)
        beta= np.zeros(sample_size-1)
        # sample[1:] = upper bounds, and vice versa
        
        tmp = sample[1:] - sample[:-1]
        tmp_logic = sample[1:]<xa
        alpha[tmp_logic] = tmp[tmp_logic]
                               
        tmp_logic = sample[:-1]>xa
        beta[tmp_logic] = tmp[tmp_logic]
        
        tmp_logic = ( sample[1:]>xa )*( sample[:-1]<xa )
        tmp = xa - sample[:-1]
        alpha[tmp_logic] = tmp[tmp_logic]
        tmp = sample[1:] - xa
        beta[tmp_logic] = tmp[tmp_logic]
        
        p = np.arange(1,sample_size)/sample_size
        c = alpha*p**2 + beta*(1-p)**2
        crps_sum = np.sum(c)
        
        # Intervals 0 and N, where p = 0 and 1 respectively
        if xa < sample[0]:
            p=0
            alpha = 0
            beta = sample[0] - xa
            crps = calc(alpha, beta, p)
            crps_sum = crps_sum + crps
        elif xa > sample[-1]:
            p=1
            alpha = xa - sample[-1]
            beta = 0
            crps = calc(alpha, beta, p)
            crps_sum = crps_sum + crps
            
        return crps_sum
    
    def get_common_x(self, other, n_pts=1000):
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
    
    def diff_plot(self, other):
        """Plots two CDFS on one plot, with the difference shaded"""
        debug(f"Generating diff plot for {get_slug(self)} and {get_slug(other)}")
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.get_common_x(other)
        dum, mod_y = self.build_discrete_cdf(x)
        ax.plot(x, mod_y, c='k', linestyle='--')
        dum, obs_y = other.build_discrete_cdf(x)
        ax.plot(x, obs_y, linestyle='--')
        ax.fill_between(x, mod_y, obs_y, alpha=0.5)
        return fig, ax
