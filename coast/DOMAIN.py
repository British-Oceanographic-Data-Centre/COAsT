from .COAsT import COAsT
from warnings import warn
import numpy as np
import xarray as xr


class DOMAIN(COAsT):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get depths at time zero
        self.set_timezero_depth()

        return

    def set_dimension_mapping(self):
        #self.dim_mapping = {'t':'t_dim', 'z':'z_dim', 
        #                    'y':'y_dim', 'x':'x_dim'}
        self.dim_mapping = None


    def find_j_i(self, lat: int, lon: int, grid_ref: str):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(49, -12, t)

        :param lat: latitude
        :param lon: longitude
        :param grid_ref: the gphi/glam version a user wishes to search over
        :return: the y and x coordinates for the given grid_ref variable within the domain file
        """

        internal_lat = f"gphi{grid_ref}"
        internal_lon = f"glam{grid_ref}"
        dist2 = xr.ufuncs.square(self.dataset[internal_lat] - lat) + xr.ufuncs.square(self.dataset[internal_lon] - lon)
        [_, y, x] = np.unravel_index(dist2.argmin(), dist2.shape)
        return [y, x]

    def transect_indices(self, start: tuple, end: tuple, grid_ref: str = 'T') -> tuple:
        """
        This methods returns the indices of a simple straight line transect.

        checks `grid_ref` has a value within (T, V, U, F) this corresponds to the gphi/glam variable a user wishes
        to use for looking up the indices from.

        :type start: tuple A lat/lon pair
        :type end: tuple A lat/lon pair
        :type grid_ref: str The gphi/glam version a user wishes to search over
        :return: array of y indices, array of x indices, number of indices in transect
        """

        assert isinstance(grid_ref, str) and grid_ref.upper() in ("T", "V", "U", "F"), \
            "grid_ref should be either \"T\", \"V\", \"U\", \"F\""


        letter = grid_ref.lower()

        [j1, i1] = self.find_j_i(start[0], start[1], letter)  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1], letter)  # lat , lon

        line_length = max(np.abs(j2 - j1), np.abs(i2 - i1)) + 1

        jj1 = [int(jj) for jj in np.round(np.linspace(j1, j2, num=line_length))]
        ii1 = [int(ii) for ii in np.round(np.linspace(i1, i2, num=line_length))]
        
        return jj1, ii1, line_length

    def subset_indices(self, start: tuple, end: tuple, grid_ref: str = 'T') -> tuple:
        """
        based off transect_indices, this method looks to return all indices between the given points.
        This results in a 'box' (Quadrilateral) of indices.
        consequently the returned lists may have different lengths.

        :param start: A lat/lon pair
        :param end: A lat/lon pair
        :param grid_ref: The gphi/glam version a user wishes to search over
        :return: list of y indices, list of x indices,
        """
        assert isinstance(grid_ref, str) and grid_ref.upper() in ("T", "V", "U", "F"), \
            "grid_ref should be either \"T\", \"V\", \"U\", \"F\""

        letter = grid_ref.lower()

        [j1, i1] = self.find_j_i(start[0], start[1], letter)  # lat , lon
        [j2, i2] = self.find_j_i(end[0], end[1], letter)  # lat , lon

        return list(np.arange(j1, j2+1)), list(np.arange(i1, i2+1))

    def set_timezero_depth(self):
        """
        Sets the depths at time zero along the vertical t and w levels. 
        Added to self.dataset.depth_t_0 and self.dataset.depth_w_0

        """
        
        depth_t = np.zeros_like( self.dataset.e3w_0 )  
        depth_t[:,0,:,:] = 0.5 * self.dataset.e3w_0[:,0,:,:]    
        depth_t[:,1:,:,:] = depth_t[:,0,:,:] + np.cumsum( self.dataset.e3w_0[:,1:,:,:], axis=1 ) 
        self.dataset['depth_t_0'] = xr.DataArray(depth_t, dims=self.dataset.e3w_0.dims)
        
        depth_w = np.zeros_like( self.dataset.e3t_0 ) 
        depth_w[:,0,:,:] = 0.0
        depth_w[:,1:,:,:] = np.cumsum( self.dataset.e3t_0, axis=1 )[:,:-1,:,:]
        self.dataset['depth_w_0'] = xr.DataArray(depth_w, dims=self.dataset.e3t_0.dims)

        return
