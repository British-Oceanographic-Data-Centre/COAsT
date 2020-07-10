import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class CDF():
    '''
    An object for storing Cumulative Distribution Function information.
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
        self.cdf_type = cdf_type
        self.cdf_func = cdf_func
        self.sample = sample
        self.sample_size = len(sample)
        self.mu       = np.nanmean(sample)
        self.sigma    = np.nanstd(sample)
        self._cdf_type_options = ['empirical', 'theoretical']
        self._cdf_func_options = ['gaussian']
        self.plot_xmin, self.plot_xmax = self.set_x_bounds()
        
    def set_x_bounds(self):
        ''' Calculate x bounds for CDF plotting '''
        # Is input a single value (st. dev == 0)
        single_value = True if self.sigma == 0 else False
        if single_value: 
            self.cdf_type = 'empirical'
        
        # Calculate x bounds as 5 std dev. either side of the mean
        if single_value:
            xmin = self.mu-1
            xmax = self.mu+1
        elif self.cdf_func == 'gaussian':
            xmin = self.mu - 5*self.sigma
            xmax = self.mu + 5*self.sigma
        return xmin, xmax
    
    def build_discrete_cdf(self, x: np.ndarray=None, n_pts: int=1000):
        """Builds a discrete CDF for plotting and direct comparison.

        Args:
            x (array): x-values over which to calculate discrete CDF values.
                       If none, these will be determined using mu and sigma.
            n_pts (int): n_pts to use for x if x=None.

        Returns:
            x and y arrays for discrete CDF.
        """
        
        # Build discrete X bounds
        if x is None:
            x = np.linspace(self.plot_xmin, self.plot_xmax, n_pts)
        
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

        return x, y
    
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
    
    def crps(self, xa):
        """Calculate CRPS as outlined in Hersbach et al. 2000

        Args:
            xa (float): A single 'observation' value which to compare against
                        CDF.

        Returns:
            A single CRPS value.
        """
        
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
        xmin = min(self.plot_xmin, other.plot_xmin)
        xmax = max(self.plot_xmax, other.plot_xmax)
        common_x = np.linspace(xmin, xmax, n_pts)
        return common_x
    
    def quick_plot(self):
        """ A quick plot showing the CDF contained in this object."""
        ax = plt.subplot(111)
        x,y = self.build_discrete_cdf()
        ax.plot(x, y)
        ax.grid()
        return
    
    def diff_plot(self, other):
        """Plots two CDFS on one plot, with the difference shaded"""
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.get_common_x(other)
        dum, mod_y = self.build_discrete_cdf(x)
        ax.plot(x, mod_y, c='k', linestyle='--')
        dum, obs_y = other.build_discrete_cdf(x)
        ax.plot(x, obs_y, linestyle='--')
        ax.fill_between(x, mod_y, obs_y, alpha=0.5)
        return fig, ax