import logging
from os import environ
from pathlib import Path
import pytest
from coast import Copernicus, Coast, Gridded, Coordinates2D


DATABASE = "nrt"
PRODUCT_ID = "global-analysis-forecast-phy-001-024"
CONFIG = (Path(__file__).parent.parent / "config" / "example_cmems_grid_t.json").resolve(strict=True)
USERNAME = environ.get("COPERNICUS_USERNAME")
PASSWORD = environ.get("COPERNICUS_PASSWORD")
CREDENTIALS = USERNAME is not None and PASSWORD is not None


@pytest.fixture(name="copernicus")
def copernicus_fixture() -> Copernicus:
    """Return a functional Copernicus data accessor."""
    return Copernicus(USERNAME, PASSWORD, DATABASE)


@pytest.fixture(name="gridded")
def gridded_fixture(copernicus) -> Gridded:
    forecast = copernicus.get_product("global-analysis-forecast-phy-001-024")
    return Gridded(fn_data=forecast, config=str(CONFIG))


@pytest.mark.skipif(condition=not CREDENTIALS, reason="Copernicus credentials are not set.")
def test_2d(gridded):
    start = Coordinates2D(10, 13)
    end = Coordinates2D(20, 50)
    # Validate test values
    assert gridded.dataset.longitude.min().item() < start.x
    assert gridded.dataset.latitude.min().item() < start.y
    assert gridded.dataset.longitude.max().item() > end.x
    assert gridded.dataset.latitude.max().item() > end.y

    # Constrain dataset
    constrained = gridded.constrain(start, end)

    # Check constrained dataset
    assert constrained.longitude.min().item() == start.x
    assert constrained.latitude.min().item() == start.y
    assert constrained.longitude.max().item() == end.x
    assert constrained.latitude.max().item() == end.y


@pytest.mark.skipif(condition=not CREDENTIALS, reason="Copernicus credentials are not set.")
def test_wrap(gridded):
    start = Coordinates2D(175, 50)
    end = Coordinates2D(gridded.dataset.longitude.max().item() + 5, 60)
    wrapped = gridded.dataset.longitude.min().item() + 5

    # Constraint dataset
    constrained = gridded.constrain(start, end)

    # Check constrained dataset
    assert wrapped < start.x
    assert wrapped in constrained.longitude
    assert constrained.max().item() == 185 % gridded.dataset.max.item()


if not CREDENTIALS:
    logging.warning("https://marine.copernicus.eu/ credentials not set, integration tests will not be run!")
