from datetime import datetime
import json
import os
from pathlib import Path
import pytest
from coast.config import ConfigParser
from coast.config.config_structure import (
    ConfigTypes,
    ConfigKeys,
    GriddedConfig,
    IndexedConfig,
    Dataset,
    Domain,
    CodeProcessing,
)

# Valid gridded config json.
gridded_json = {
    "type": "gridded",
    "grid_ref": {},
    "dimensionality": 3,
    "chunks": [],
    "dataset": {"variable_map": {}, "dimension_map": {}},
    "domain": {
        "variable_map": {},
        "dimension_map": {},
    },
    "static_variables": {"not_grid_vars": [], "coord_vars": [], "delete_vars": []},
    "processing_flags": [],
}

# Valid indexed config json.
indexed_json = {
    "type": "indexed",
    "dimensionality": 3,
    "chunks": [],
    "dataset": {"variable_map": {}, "dimension_map": {}},
    "processing_flags": [],
}

# Json with an invalid type value.
invalid_type_json = {"type": "invalid"}


@pytest.fixture
def json_file(input_json):
    """Write example json to file, for use with ConfigParser(). File auto deleted after test."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    tempfile = Path(f"{dir_path}/tempfile_{cur_time}.json")
    with open(tempfile, "w") as temp:
        json.dump(input_json, temp)
    yield tempfile
    tempfile.unlink()


def test__parse_gridded():
    """Test _parse_gridded method doesn't error on valid gridded json."""
    gridded_obj = ConfigParser._parse_gridded(gridded_json)
    assert type(gridded_obj) is GriddedConfig
    assert gridded_obj.type is ConfigTypes.GRIDDED


def test__parse_indexed():
    """Test _parse_indexed method doesn't error on valid indexed json."""
    indexed_obj = ConfigParser._parse_indexed(indexed_json)
    assert type(indexed_obj) is IndexedConfig
    assert indexed_obj.type is ConfigTypes.INDEXED


@pytest.mark.parametrize(
    "config_json, object_key, object_type",
    [
        (gridded_json, ConfigKeys.DATASET, Dataset),
        (gridded_json, ConfigKeys.DOMAIN, Domain),
    ],
)
def test__get_datafile_object(config_json, object_key, object_type):
    """Test _get_datafile_object method for both Dataset and Domain."""
    data_obj = ConfigParser._get_datafile_object(config_json, object_key)
    assert type(data_obj) is object_type


@pytest.mark.parametrize(
    "config_json, , object_type",
    [
        (gridded_json, CodeProcessing),
    ],
)
def test__get_code_processing_object(config_json, object_type):
    """Test _get_code_processing_object method."""
    data_obj = ConfigParser._get_code_processing_object(config_json)
    assert type(data_obj) is object_type


# input_json argument indirectly links to json_file(input_json) method argument.
@pytest.mark.parametrize(
    "input_json, config_class, config_type",
    [
        (gridded_json, GriddedConfig, ConfigTypes.GRIDDED),
        (indexed_json, IndexedConfig, ConfigTypes.INDEXED),
    ],
)
def test_config_parser(json_file, config_class, config_type):
    """Test config parser init method with valid gridded and indexed json."""
    config = ConfigParser(json_file).config
    assert type(config) is config_class
    assert config.type is config_type


@pytest.mark.parametrize("input_json", [invalid_type_json])
def test_config_parser_invalid_type(json_file):
    """Test config parser with an invalid type in json."""
    with pytest.raises(ValueError) as e:
        config = ConfigParser(json_file)
