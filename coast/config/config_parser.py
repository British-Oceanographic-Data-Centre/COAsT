"""Config parser."""
import json
from pathlib import Path
from typing import Union

from .config_structure import ConfigTypes, ConfigKeys, GriddedConfig, IndexedConfig, Dataset, Domain, CodeProcessing


class ConfigParser:
    """Class for parsing gridded and indexed configuration files."""

    def __init__(self, json_path: Union[Path, str]):
        """Config parser constructor.

        Args:
            json_path (Union[Path, str]): path to json config file.
        """
        with open(json_path, "r") as j:
            json_content = json.loads(j.read())
            conf_type = ConfigTypes(json_content[ConfigKeys.TYPE])
            if conf_type == ConfigTypes.GRIDDED:
                self.config = ConfigParser._parse_gridded(json_content)
            elif conf_type == ConfigTypes.INDEXED:
                self.config = ConfigParser._parse_indexed(json_content)

    @staticmethod
    def _parse_gridded(json_content: dict) -> GriddedConfig:
        """Static method to parse Gridded config files.

        Args:
            json_content (dict): Config file json.
        """
        dimensionality = json_content[ConfigKeys.DIMENSIONALITY]
        grid_ref = json_content[ConfigKeys.GRID_REF]
        proc_flags = json_content[ConfigKeys.PROC_FLAGS]
        chunks = json_content[ConfigKeys.CHUNKS]
        dataset = ConfigParser._get_datafile_object(json_content, ConfigKeys.DATASET)
        static_variables = ConfigParser._get_code_processing_object(json_content)
        try:
            domain = ConfigParser._get_datafile_object(json_content, ConfigKeys.DOMAIN)
        except KeyError:
            domain = None
        return GriddedConfig(
            dimensionality=dimensionality,
            grid_ref=grid_ref,
            chunks=chunks,
            dataset=dataset,
            domain=domain,
            processing_flags=proc_flags,
            code_processing=static_variables,
        )

    @staticmethod
    def _parse_indexed(json_content: dict) -> IndexedConfig:
        """Static method to parse Indexed config files.

        Args:
            json_content (dict): Config file json.
        """
        dimensionality = json_content[ConfigKeys.DIMENSIONALITY]
        proc_flags = json_content[ConfigKeys.PROC_FLAGS]
        chunks = json_content[ConfigKeys.CHUNKS]
        dataset = ConfigParser._get_datafile_object(json_content, ConfigKeys.DATASET)
        return IndexedConfig(dimensionality=dimensionality, chunks=chunks, dataset=dataset, processing_flags=proc_flags)

    @staticmethod
    def _get_code_processing_object(json_content: dict) -> CodeProcessing:
        """Static method to convert static_variables configs into objects.

        Args:
            json_content (dict): Config file json.
        """
        dataset_json = json_content[ConfigKeys.CODE_PROCESSING]
        return CodeProcessing(
            delete_variables=dataset_json[ConfigKeys.DEL_VAR],
            not_grid_variables=dataset_json[ConfigKeys.NO_GR_VAR],
        )

    @staticmethod
    def _get_datafile_object(json_content: dict, data_file_type: str) -> Union[Dataset, Domain]:
        """Static method to convert dataset and domain configs into objects.

        Args:
            json_content (dict): Config file json.
            data_file_type (str): key of datafile type (dataset or domain).
        """
        dataset_json = json_content[data_file_type]
        dataset_var = dataset_json[ConfigKeys.VAR_MAP]
        dataset_dim = dataset_json[ConfigKeys.DIM_MAP]

        try:
            dataset_keep_all_vars = dataset_json[ConfigKeys.KEEP_ALL_VARS]
        except KeyError:
            dataset_keep_all_vars = "False"

        if data_file_type is ConfigKeys.DATASET:
            dataset_coord_vars = dataset_json[ConfigKeys.COO_VAR]
            return Dataset(
                variable_map=dataset_var,
                dimension_map=dataset_dim,
                coord_var=dataset_coord_vars,
                keep_all_vars=dataset_keep_all_vars,
            )
        elif data_file_type is ConfigKeys.DOMAIN:
            return Domain(variable_map=dataset_var, dimension_map=dataset_dim, keep_all_vars=False)
