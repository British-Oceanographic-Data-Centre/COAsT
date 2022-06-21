from pathlib import Path
from os import environ
import pytest
from coast import Copernicus, Product, Gridded

DATABASE = "nrt"
PRODUCT_ID = "global-analysis-forecast-phy-001-024"
CONFIG = (Path(__file__).parent.parent / "config" / "example_cmems_grid_t_DEV.json").resolve(strict=True)

USERNAME = environ.get("COPERNICUS_USERNAME")
PASSWORD = environ.get("COPERNICUS_PASSWORD")


@pytest.fixture(name="copernicus")
def copernicus_fixture():
    return Copernicus(USERNAME, PASSWORD, DATABASE)


@pytest.mark.skipif(condition=not USERNAME or not PASSWORD,  reason="Copernicus credentials are not set.")
def test_get_product(copernicus):
    product = copernicus.get_product(PRODUCT_ID)
    nemo_t = Gridded(fn_data=product, config=str(CONFIG))

    dimensions = nemo_t.dataset.dims

    # Verify that dimensions have been mapped correctly
    assert "x_dim" in dimensions
    assert "y_dim" in dimensions
    assert "z_dim" in dimensions
    assert "t_dim" in dimensions
