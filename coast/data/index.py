"""Index class."""
from dask import array
from dask.distributed import Client
from .._utils.logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
from ast import literal_eval
from .coast import Coast
from .config_parser import ConfigParser


def setup_dask_client(workers: int = 2, threads: int = 2, memory_limit_per_worker: str = "2GB"):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class Indexed(Coast):
    def __init__(self, config: Union[Path, str] = None):
        """Configuration init.

        Args:
            config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")

        self.dataset = None
        self.chunks = None
        self.var_mapping = None
        self.dim_mapping = None
        self.coord_vars = None
        self.keep_all_vars = False

        if config:
            print(config)
            self.json_config = ConfigParser(config)
            self.chunks = self.json_config.config.chunks
            self.dim_mapping = self.json_config.config.dataset.dimension_map
            self.var_mapping = self.json_config.config.dataset.variable_map
            self.coord_vars = self.json_config.config.dataset.coord_var
            self.keep_all_vars = literal_eval(self.json_config.config.dataset.keep_all_vars)

    def apply_config_mappings(self) -> None:
        """Applies json configuration and mappings"""
        # We iterate through each mapping one by one which enables us to catch those variables that do not exist

        if self.dim_mapping is not None:
            for k in self.dim_mapping:
                try:
                    self.dataset = self.dataset.rename_dims({k: self.dim_mapping[k]})
                except ValueError as e:
                    debug(f"Warning: {str(e)}")

        if self.var_mapping is not None:
            keep_vars = []
            for k in self.var_mapping:
                try:
                    self.dataset = self.dataset.rename_vars({k: self.var_mapping[k]})
                    keep_vars.append(self.var_mapping[k])
                except ValueError as e:
                    debug(f"Warning: {str(e)}")

            if not self.keep_all_vars:
                self.dataset = self.dataset[keep_vars]

        if self.coord_vars is not None:
            self.dataset = self.dataset.set_coords(self.coord_vars)

    def insert_dataset(self, dataset, apply_config_mappings=False):
        """Insert a dataset straight into this object instance"""
        self.dataset = dataset
        if apply_config_mappings:
            self.apply_config_mappings()
