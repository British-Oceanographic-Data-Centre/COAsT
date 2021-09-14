"""Classes defining config structure."""
from dataclasses import dataclass
from enum import Enum, unique


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
    """
    pass


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
        chunks (tuple): Tuple for dask chunking config. (i.e. (1000,1000,1000)).
        type (ConfigTypes): Type of config. Must be a valid ConfigType.
    """
    dimensionality: int
    dataset: Dataset
    processing_flags: list
    chunks: tuple
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
