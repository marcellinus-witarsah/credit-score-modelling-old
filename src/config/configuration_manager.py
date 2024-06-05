from pathlib import Path
from src.utils.common import read_yaml
from src.utils.common import create_directories
from src.constants import CONFIG_FILE_PATH
from src.constants import PARAMS_FILE_PATH
from src.constants import SCHEMA_FILE_PATH
from src.entities.config_entity import DataIngestionConfig
from src.entities.config_entity import DataValidationConfig


class ConfigurationManager:
    """
    Prepare ConfigurationManager class.

    This class is responsible for reading configuration files and preparing
    configuration settings for the pipeline.

    Attributes:
        config (dict): Parsed configuration file content.
        params (dict): Parsed parameters file content.
        schema (dict): Parsed schema file content.
    """

    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
        schema_filepath: str = SCHEMA_FILE_PATH,
    ):
        """
        Initialize the ConfigurationManager with file paths.

        Args:
            config_filepath (str): File path to the configuration YAML file.
            params_filepath (str): File path to the parameters YAML file.
            schema_filepath (str): File path to the schema YAML file.
        """
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))
        self.schema = read_yaml(Path(schema_filepath))
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get configuration for data ingestion.

        Returns:
            DataIngestionConfig: Configuration for data ingestion.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_path=Path(config.source_path),
            target_path=Path(config.target_path),
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get configuration for data validation.

        Returns:
            DataValidationConfig: Configuration for data validation.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            source_path=config.source_path,
            STATUS_FILE=config.STATUS_FILE,
            schema=schema,
        )
        return data_validation_config
