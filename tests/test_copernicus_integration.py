"""Integration tests for Copernicus functionality."""

import logging
from pathlib import Path
from os import environ
import pytest
import numpy
from xarray import Dataset
from coast import Copernicus, Gridded


DATABASE = "nrt"
PRODUCT_ID = "global-analysis-forecast-phy-001-024"
CONFIG = (Path(__file__).parent.parent / "config" / "example_cmems_grid_t.json").resolve(strict=True)

USERNAME = environ.get("COPERNICUS_USERNAME")
PASSWORD = environ.get("COPERNICUS_PASSWORD")

CREDENTIALS = USERNAME is not None and PASSWORD is not None

if not CREDENTIALS:
    logging.warning("https://marine.copernicus.eu/ credentials not set, integration tests will not be run!")


@pytest.fixture(name="copernicus")
def copernicus_fixture() -> Copernicus:
    """Return a functional Copernicus data accessor."""
    return Copernicus(USERNAME, PASSWORD, DATABASE)


@pytest.mark.skipif(condition=not CREDENTIALS, reason="Copernicus credentials are not set.")
def test_get_product(copernicus):
    """Connect to Copernicus, access some metadata and data values and ingest into a Gridded object."""
    product = copernicus.get_product(PRODUCT_ID)
    nemo_t = Gridded(fn_data=product, config=str(CONFIG))

    # Check that the dataset has been initialised as expected
    assert isinstance(nemo_t.dataset, Dataset)
    # Check for some expected CMEMS metadata
    assert nemo_t.dataset.attrs["comment"] == "CMEMS product"
    assert nemo_t.dataset.attrs["easting"] == "longitude"
    assert nemo_t.dataset.attrs["northing"] == "latitude"
    assert nemo_t.dataset.attrs["institution"] == "MERCATOR OCEAN"
    # Verify that dimensions have been mapped correctly
    dimensions = nemo_t.dataset.dims
    assert "x_dim" in dimensions
    assert "y_dim" in dimensions
    assert "z_dim" in dimensions
    assert "t_dim" in dimensions
    # Download and inspect some values
    latitude = nemo_t.dataset.latitude[:-100].values
    assert all(isinstance(value, numpy.float32) and not numpy.isnan(value) for value in latitude)
