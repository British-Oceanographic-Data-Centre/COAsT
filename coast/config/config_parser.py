"""Config parser."""
import json

from .config_structure import (
    ConfigTypes,
    ConfigKeys,
    GriddedConfig,
    IndexedConfig,
    Dataset,
    Domain
)

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