"""Functionality for accessing OPeNDAP datasets."""

from dataclasses import dataclass
from typing import Optional
from xarray.backends import PydapDataStore
from xarray import Dataset, open_dataset
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import requests


CASTGC = "CASTGC"


@dataclass
class OpendapInfo:
    """A class for accessing streamable OPeNDAP data."""

    url: str
    session: Optional[requests.Session] = None

    def get_store(self) -> PydapDataStore:
        """Access an OPeNDAP data store.

        Returns:
            The OPeNDAP data store accessed from the instance's URL.
        """
        return PydapDataStore(open_url(self.url, session=self.session))

    def open_dataset(self, chunks: Optional[dict] = None) -> Dataset:
        """Open the remote XArray dataset for streaming.

        Args:
            chunks: Chunks to use in Dask.

        Returns:
            The opened XArray dataset.
        """
        with open_dataset(self.get_store(), chunks=chunks) as dataset:
            return dataset

    @classmethod
    def from_cas(cls, url: str, cas_url: str, username: str, password: str) -> "OpendapInfo":
        """Instantiate OpendapInfo with a session authenticated against CAS.

        Args:
            url: The OPeNDAP dataset URL.
            cas_url: The CAS login URL.
            username: The username to authenticate with.
            password: The password to authenticate with.

        Returns:
            The instantiated OPeNDAP accessor.
        """
        session = setup_session(cas_url, username, password)
        session.cookies.set(CASTGC, session.cookies.get_dict()[CASTGC])
        return cls(url, session)
