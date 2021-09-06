from datetime import datetime
import json
import os
from pathlib import Path
import pytest
from coast.config import ConfigParser
from coast.config.config_structure import ConfigTypes, GriddedConfig, IndexedConfig

@pytest.fixture
def gridded_json():
    """Bare minimum for valid gridded json."""
    return {
        "type": "gridded",
        "grid_ref":{},
        "dimensionality":3,
        "dataset":{
            "variable_map":{},
            "dimension_map":{},
            "chunks": []
        },
        "domain":{
            "variable_map":{},
            "dimension_map":{},
        },
        "processing_flags":[]
    }

@pytest.fixture
def indexed_json():
    """Bare minimum for valid indexed json."""
    return {
        "type": "indexed",
        "dimensionality":3,
        "dataset":{
            "variable_map":{},
            "dimension_map":{},
            "chunks":[]
        },
        "processing_flags":[]
    }


@pytest.fixture
def gridded_json_file(gridded_json):
    """Write example json to file, for use with ConfigParser()."""
    tempfile = _get_temp_name()
    with open(tempfile, 'w') as temp:
        json.dump(gridded_json, temp)
    yield tempfile
    tempfile.unlink()


@pytest.fixture
def indexed_json_file(indexed_json):
    """Write example json to file, for use with ConfigParser()."""
    tempfile = _get_temp_name()
    with open(tempfile, 'w') as temp:
        json.dump(indexed_json, temp)
    yield tempfile
    tempfile.unlink()


def _get_temp_name():
    """Helper method to return a temp file name."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(f"{dir_path}/tempfile_{cur_time}.json")


def test__parse_gridded(gridded_json):
    """Test parse gridded method doesn't error on valid gridded json."""
    gridded_obj = ConfigParser._parse_gridded(gridded_json)
    assert type(gridded_obj) is GriddedConfig
    assert gridded_obj.type is ConfigTypes.GRIDDED


def test__parse_indexed(indexed_json):
    """Test parse indexed method doesn't error on valid indexed json."""
    indexed_obj = ConfigParser._parse_indexed(indexed_json)
    assert type(indexed_obj) is IndexedConfig
    assert indexed_obj.type is ConfigTypes.INDEXED


def test_config_parser_gridded(gridded_json_file):
    configparser = ConfigParser(gridded_json_file)
    config = configparser.config
    assert type(config) is GriddedConfig
    assert config.type is ConfigTypes.GRIDDED


def test_config_parser_indexed(indexed_json_file):
    configparser = ConfigParser(indexed_json_file)
    config = configparser.config
    assert type(config) is IndexedConfig
    assert config.type is ConfigTypes.INDEXED


