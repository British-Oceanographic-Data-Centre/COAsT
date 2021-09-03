"""Classes related to data file Configuration"""

from dataclasses import dataclass
from enum import Enum, unique
import json


@unique
class ConfigTypes(Enum):
    """Enum class containing the valid types for config files."""
    GRIDDED = "gridded"
    INDEXED = "indexed"


class ConfigKeys():
    """Class of constants representing valid keys within configuriation json."""
    TYPE="type"
    DIMENSIONALITY="dimensionality"
    GRIDREF="grid_ref"
    PROC_FLAGS="processing_flags"
    DATASET="dataset"
    DOMAIN="domain"
    DIM_MAP="dimension_map"
    VAR_MAP="variable_map"
    CHUNKS="chunks"


class ConfigParser():
    """Class for parsing gridded and indexed configuration files."""
    def __init__(self, json_path):
        """Config parser constructor.
        
        Args:
            json_path (str): path to json config file.
        """
        with open(json_path, 'r') as j:
            json_content = json.loads(j.read())
            conf_type = ConfigTypes(json_content[ConfigKeys.TYPE])
            if conf_type == ConfigTypes.GRIDDED:
                self.config = ConfigParser._parse_gridded(json_content)
            elif conf_type == ConfigTypes.INDEXED:
                self.config = ConfigParser._parse_indexed(json_content)


    @staticmethod
    def _parse_gridded(json_content):
        """Static method to parse Gridded config files.
        
        Args:
            json_content (dict): Config file json.
        """
        dimensionality = json_content[ConfigKeys.DIMENSIONALITY]
        grid_ref = json_content[ConfigKeys.GRIDREF]
        proc_flags = json_content[ConfigKeys.PROC_FLAGS]
        dataset = ConfigParser._get_datafile_object(json_content, ConfigKeys.DATASET)
        try:
            domain = ConfigParser._get_datafile_object(json_content, ConfigKeys.DOMAIN)
        except KeyError:
            domain = None
        return GriddedConfig(
            dimensionality=dimensionality, grid_ref=grid_ref, dataset=dataset, domain=domain, processing_flags=proc_flags
            )


    @staticmethod
    def _parse_indexed(json_content):
        """Static method to parse Indexed config files.
        
        Args:
            json_content (dict): Config file json.
        """
        dimensionality = json_content[ConfigKeys.DIMENSIONALITY]
        proc_flags = json_content[ConfigKeys.PROC_FLAGS]
        dataset = ConfigParser._get_datafile_object(json_content, ConfigKeys.DATASET)
        return IndexedConfig(
            dimensionality=dimensionality ,dataset=dataset, processing_flags=proc_flags
            )


    @staticmethod
    def _get_datafile_object(json_content, data_file_type):
        """Static method to convert dataset and domain configs into objects.
        
            Args:
                json_content (dict): Config file json.
                data_file_type (str): key of datafile type (dataset or domain).
        """
        dataset_json = json_content[data_file_type]
        dataset_var = dataset_json[ConfigKeys.VAR_MAP]
        dataset_dim = dataset_json[ConfigKeys.DIM_MAP]

        if data_file_type is ConfigKeys.DATASET:
            chunks = tuple(dataset_json[ConfigKeys.CHUNKS])
            return Dataset(variable_map= dataset_var, dimension_map=dataset_dim, chunks=chunks)
        elif data_file_type is ConfigKeys.DOMAIN:
            return Domain(variable_map=dataset_var, dimension_map=dataset_dim)


@dataclass(frozen=True)
class DataFile():
    """General parent dataclass for holding common config attributes of datafiles.
    
    Args:
        variable_map (dict): dict containing mapping for variable names.
        dimension_map (dict): dict containing mapping for dimension names.
    """
    variable_map: dict
    dimension_map: dict


@dataclass(frozen=True)
class Dataset(DataFile):
    """
    Dataclass holding config attributes for Dataset datafiles. Extends DataFile.

    Args:
        chunks (tuple): Tuple for dask chunking config. (i.e. (1000,1000,1000)).
    """
    chunks: tuple


@dataclass(frozen=True)
class Domain(DataFile):
    """
    Dataclass holding config attributes for Domain datafiles. Extends DataFile.
    """
    pass


@dataclass(frozen=True)
class Config():
    """General dataclass for holding common config file attributes.
    
    Args:
        dataset (Dataset): Dataset object representing 'dataset' config.
        processing_flags (list): List of processing flags.
        type (ConfigTypes): Type of config. Must be a valid ConfigType.
    """
    dimensionality: int
    dataset: Dataset
    processing_flags: list
    type: ConfigTypes


@dataclass(frozen=True)
class GriddedConfig(Config):
    """Dataclass for holding gridded-config specific attributes. Extends Config.
    
    Args:
        type (ConfigTypes): Type of config. Set to ConfigTypes.GRIDDED.
        grid_ref (dict): dict containing key:value of grid_ref:[list of grid variables].
        domain (Domain): Domain object representing 'domain' config.
    """
    type: ConfigTypes = ConfigTypes.GRIDDED
    grid_ref: dict = None
    domain: Domain = None


@dataclass(frozen=True)
class IndexedConfig(Config):
    """Dataclass for holding indexed-config specific attributes. Extends Config.
    
    Args:
        type (ConfigTypes): Type of config. Set to ConfigTypes.INDEXED.
    """
    type: ConfigTypes = ConfigTypes.INDEXED
