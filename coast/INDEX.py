"""WIP: INDEX class."""
from dask import array
from dask.distributed import Client
from .logging_util import get_slug, debug, info, warn, warning
from .config import config_parser, config_structure
from .logging_util import get_slug, debug, info, warn, warning
from typing import Union
from pathlib import Path
from ast import literal_eval


def setup_dask_client(
        workers: int = 2,
        threads: int = 2,
        memory_limit_per_worker: str = '2GB'
):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class INDEXED:
    def __init__(self, config: Union[Path, str] = None):
        """ Configuration init.

            Args:
                config (Union[Path, str]): path to json config file.
        """
        debug(f"Creating a new {get_slug(self)}")

        self.dataset = None
        self.chunks = None
        self.var_mapping = None
        self.dim_mapping = None
        self.load_all = True

        if config:
            self.json_config = config_parser.ConfigParser(config)
            if self.json_config.config.chunks:
                self.chunks = literal_eval(self.json_config.config.chunks[0])
            self.dim_mapping = self.json_config.config.dataset.dimension_map
            self.var_mapping = self.json_config.config.dataset.variable_map
            # self.load_all = self.json_config.config.dataset.load_all

    def apply_config_mappings(self):
        """Applies json configuration mappings"""
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

            # Just keep the variables specified in the json config variable mapping
            if self.load_all:
                self.dataset = self.dataset[keep_vars]