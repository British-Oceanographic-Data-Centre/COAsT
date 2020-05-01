from .COAsT import COAsT
from dask import array


class SUBSET(COAsT):

    def __init__(self, domain: COAsT, nemo: COAsT):
        self.domain_data_obj = domain
        self.nemo_data_obj = nemo

    def subset(self, points_a: array, points_b: array):
        self.dataset = super().transect(self.domain_data_obj, self.nemo_data_obj, points_a, points_b)

    def get_density(self):
        raise NotImplementedError

    def get_nearest_neighbour(self):
        raise NotImplementedError
