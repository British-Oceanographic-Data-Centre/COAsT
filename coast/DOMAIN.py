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
        self.dim_mapping = {'t':'t_dim', 'z':'z_dim',
                            'y':'y_dim', 'x':'x_dim'}
        #self.dim_mapping = None

    def construct_depths_from_spacings(self):
        # NEMO4 constucts depths from verical spacing variables
        self.depth_t = xr.DataArray( self.dataset.e3t_0.cumsum( dim='z_dim' ).squeeze() ) # size: nz,my,nx
        self.depth_t.attrs['units'] = 'm'
        self.depth_t.attrs['standard_name'] = 'depth_at_t-points'

        self.depth_w = xr.DataArray( self.dataset.e3w_0.cumsum( dim='z_dim' ).squeeze() ) # size: nz,my,nx
        self.depth_w.attrs['units'] = 'm'
        self.depth_w.attrs['standard_name'] = 'depth_at_w-points'

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


