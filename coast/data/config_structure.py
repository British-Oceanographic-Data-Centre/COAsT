"""Classes defining config structure."""
from dataclasses import dataclass
from enum import Enum, unique


@unique
class ConfigTypes(Enum):
    """Enum class containing the valid types for config files."""

    GRIDDED = "gridded"
    INDEXED = "indexed"


class ConfigKeys:
    """Class of constants representing valid keys within configuriation json."""

    TYPE = "type"
    DIMENSIONALITY = "dimensionality"
    GRID_REF = "grid_ref"
    PROC_FLAGS = "processing_flags"
    DATASET = "dataset"
    DOMAIN = "domain"
    CODE_PROCESSING = "static_variables"
    DIM_MAP = "dimension_map"
    VAR_MAP = "variable_map"
    CHUNKS = "chunks"
    NO_GR_VAR = "not_grid_vars"
    COO_VAR = "coord_vars"
    DEL_VAR = "delete_vars"
    KEEP_ALL_VARS = "keep_all_vars"


@dataclass(frozen=True)
class DataFile:
    """General parent dataclass for holding common config attributes of datafiles.

    Args:
        variable_map (dict): dict containing mapping for variable names.
        dimension_map (dict): dict containing mapping for dimension names.
        keep_all_vars (boolean): True if xarray is to retain all data file variables
                                  otherwise False i.e keep only those in the json config file variable mappings.
    """

    variable_map: dict
    dimension_map: dict
    keep_all_vars: bool = False


@dataclass(frozen=True)
class CodeProcessing:
    """Dataclass holding config attributes for static variables that might not need changing between model runs

    Args:
        not_grid_variables (list): A list of variables not belonging to the grid.
        delete_variables (list):  A list of variables to drop from the dataset.
    """

    not_grid_variables: list
    delete_variables: list


@dataclass(frozen=True)
class Dataset(DataFile):
    """Dataclass holding config attributes for Dataset datafiles. Extends DataFile.

    Args:
        coord_var (list): list of dataset coordinate variables to apply once dataset is loaded
    """

    coord_var: list = None
    pass


@dataclass(frozen=True)
class Domain(DataFile):
    """Dataclass holding config attributes for Domain datafiles. Extends DataFile."""

    pass


@dataclass(frozen=True)
class Config:
    """General dataclass for holding common config file attributes.

    Args:
        dataset (Dataset): Dataset object representing 'dataset' config.
        processing_flags (list): List of processing flags.
        chunks (dict): dict for dask chunking config. (i.e. {"dim1":100, "dim2":100, "dim3":100}).
        type (ConfigTypes): Type of config. Must be a valid ConfigType.
    """

    dimensionality: int
    dataset: Dataset
    processing_flags: list
    chunks: dict
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
    code_processing: CodeProcessing = None


@dataclass(frozen=True)
class IndexedConfig(Config):
    """Dataclass for holding indexed-config specific attributes. Extends Config.

    Args:
        type (ConfigTypes): Type of config. Set to ConfigTypes.INDEXED.
    """

    type: ConfigTypes = ConfigTypes.INDEXED
