"""

Test loading various types of INDEXED datasets with and without configuration
 files

        pytest -s tests/test_index_classes_load_datasets.py

This file was in directory tests/ and executed under Git Actions. However, it
calls external data that are too large for accessible hosting. Nevertheless,
the structure proposed here "one file per type" could lend itself well to a
restructure unit_testing work flow. So I move it there as a potential template
for when the unit testing is restructured.
"""
from coast import Altimetry, Profile, Glider, Argos, Oceanparcels, Tidegauge
import datetime

# Altimetry data (NetCDF)
fn_altimetry = "./example_files/COAsT_example_altimetry_data.nc"
fn_altimetry_config = "config/example_altimetry.json"

# En4 profile data (NetCDF)
fn_profile = "./example_files/EN.4.2.1.f.profiles.l09.201501.nc"
fn_profile_config = "config/example_en4_profiles.json"

# EGO glider (NetCDF). More examples see https://www.bodc.ac.uk/data/bodc_database/gliders/
fn_glider = "./example_files/Doombar_553_R.nc"
fn_glider_config = "config/example_glider_ego.json"

# Argos drifter data (CSV format)
fn_argos = "./example_files/ARGOS.CSV"
fn_argos_config = "config/example_argos.json"

# Ocean parcels data (NetCDF)
fn_oceanparc = "./example_files/OceanParcels_GFDL-ESM2M.2043r3.1.nc"
fn_oceanparc_config = "config/example_oceanparcels.json"

# Tidegauge data GESLA file (format version 3.0)
fn_tidegauge = "./example_files/tide_gauges/portellen-p202-uk-bodc"
fn_tidegauge_config = "config/example_tidegauge.json"
fn_tidegauge_multiple = "./example_files/tide_gauges/m*"

# wod profile data 1D (NetCDF)
fn_wod = "./example_files/wod_example_ragged_standard_level.nc"
fn_wod_config = "config/example_wod_profiles.json"


def test_load_altimetry_no_config():
    altimetry = Altimetry(file_path=fn_altimetry)
    assert altimetry is not None


def test_load_altimetry_config():
    altimetry = Altimetry(file_path=fn_altimetry, config=fn_altimetry_config)
    assert altimetry is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(altimetry.dataset.coords) == ["time", "longitude", "latitude"]
    assert list(altimetry.dataset.data_vars) == ["ocean_tide_standard_name"]


def test_load_profile_no_config():
    profile = Profile(file_path=fn_altimetry)
    assert profile is not None


def test_load_profile_config():
    profile = Profile(file_path=fn_profile, config=fn_profile_config)
    assert profile is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(profile.dataset.coords) == ["latitude", "longitude", "time"]
    assert list(profile.dataset.data_vars) == [
        "depth",
        "temperature",
        "qc_potential_temperature",
        "qc_practical_salinity",
        "qc_depth",
        "qc_time",
    ]


def test_load_wod_config():
    wod_profile_1D = Profile(config=fn_wod_config)
    wod_profile_1D.read_wod(fn_wod)
    assert wod_profile_1D is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(wod_profile_1D.dataset.coords) == ["casts", "Z_N"]
    assert list(wod_profile_1D.dataset.data_vars) == [
        "depth",
    ]


def test_load_glider_no_config():
    glider = Glider(file_path=fn_glider)
    assert glider is not None


def test_load_glider_config():
    glider = Glider(file_path=fn_glider, config=fn_glider_config)
    assert glider is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(glider.dataset.coords) == ["TIME", "TIME_GPS", "LATITUDE", "LONGITUDE", "PRES"]
    assert list(glider.dataset.data_vars) == [
        "PLATFORM_TYPE",
        "PLATFORM_MAKER",
        "LATITUDE_GPS",
        "LONGITUDE_GPS",
        "TEMP",
    ]


def test_load_argos_no_config():
    argos = Argos(file_path=fn_argos)
    assert argos is not None


def test_load_argos_config():
    argos = Argos(file_path=fn_argos, config=fn_argos_config)
    assert argos is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(argos.dataset.coords) == ["time", "latitude", "longitude"]
    assert list(argos.dataset.data_vars) == ["sea_surface_temperature", "pressure"]


def test_load_oceanparcels_no_config():
    oceanparc = Oceanparcels(file_path=fn_oceanparc)
    assert oceanparc is not None


def test_load_oceanparcels_config():
    oceanparc = Oceanparcels(file_path=fn_oceanparc, config=fn_oceanparc_config)
    assert oceanparc is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(oceanparc.dataset.coords) == ["time", "latitude", "longitude"]
    assert list(oceanparc.dataset.data_vars) == ["depth", "t", "s"]


def test_load_tidegauge_no_config():
    tide_gauge = Tidegauge(file_path=fn_tidegauge)
    assert tide_gauge is not None


def test_load_tidegauge_config():
    date_0 = datetime.datetime(2007, 1, 10)
    date_1 = datetime.datetime(2007, 1, 12)

    tide_gauge = Tidegauge(file_path=fn_tidegauge, date_start=date_0, date_end=date_1, config=fn_tidegauge_config)
    assert tide_gauge is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(tide_gauge.dataset.coords) == ["time"]
    assert list(tide_gauge.dataset.data_vars) == ["h", "qc_flags", "lat_dim", "lon_dim"]


def test_load_tidegauge_multiple_config():
    date_0 = datetime.datetime(2007, 1, 10)
    date_1 = datetime.datetime(2007, 1, 12)

    tide_gauge_list = Tidegauge.create_multiple_tidegauge(
        file_list=fn_tidegauge_multiple, date_start=date_0, date_end=date_1, config=fn_tidegauge_config
    )

    assert tide_gauge_list is not None
    # assert that we have the coordinate and data variable names as specified in the json config file
    assert list(tide_gauge_list[0].dataset.coords) == ["time"]
    assert list(tide_gauge_list[0].dataset.data_vars) == ["h", "qc_flags", "lat_dim", "lon_dim"]
