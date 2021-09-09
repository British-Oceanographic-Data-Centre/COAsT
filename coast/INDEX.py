"""Parent class for PROFILE, TIDEGAUGE and ALTIMETRY."""
from dask import array
import xarray as xr
import numpy as np
from dask.distributed import Client
import copy
from .logging_util import get_slug, debug, info, warn, warning


def setup_dask_client(
        workers: int = 2,
        threads: int = 2,
        memory_limit_per_worker: str = '2GB'
):
    Client(n_workers=workers, threads_per_worker=threads, memory_limit=memory_limit_per_worker)


class INDEXED:
    """Contains common sub-setting methods used by the PROFILE, TIDEGAUGE and ALTIMETRY sub classes."""

    def __init__(
            self
    ):
        debug(f"INDEX Creating a new {get_slug(self)}")
