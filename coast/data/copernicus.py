"""Functionality for accessing Copernicus datasets via OPeNDAP."""

from dataclasses import dataclass
from .opendap import OpendapInfo


COPERNICUS_CAS = "https://cmems-cas.cls.fr/cas/login"
COPERNICUS_URL = "https://{}.cmems-du.eu/thredds/dodsC/{}"


@dataclass
class CopernicusBase:
    """Information required for accessing Copernicus datasets via OPeNDAP."""

    username: str
    password: str
    database: str
    cas_url: str = COPERNICUS_CAS
    url_template: str = COPERNICUS_URL

    def get_url(self, product_id: str):
        """Get the URL required to access a Copernicus OPeNDAP dataset.

        Args:
            product_id: The product ID belonging to the chosen dataset.

        Returns:
            The constructed URL.
        """
        return self.url_template.format(self.database, product_id)


class Product(OpendapInfo):
    """Information required to access and stream data from a Copernicus product."""

    @classmethod
    def from_copernicus(cls, product_id: str, copernicus: CopernicusBase) -> "Product":
        """Instantiate a Product using Copernicus information and a specific product ID.

        Args:
            product_id: The product ID of the chosen Copernicus OPeNDAP dataset.
            copernicus: A previously instantiated Copernicus info object.

        Returns:
            An instantiated Product accessor.
        """
        return cls.from_cas(
            copernicus.get_url(product_id), copernicus.cas_url, copernicus.username, copernicus.password
        )


class Copernicus(CopernicusBase):
    """An object for accessing Copernicus products via OPeNDAP."""

    def get_product(self, product_id: str) -> Product:
        """Instantiate a Product related to a specific product ID.

        Args:
            product_id: The product ID of the chosen Copernicus OPeNDAP dataset.

        Returns:
            The instantiated Product accessor.
        """
        return Product.from_copernicus(product_id, self)
