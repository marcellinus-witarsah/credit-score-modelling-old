from pathlib import Path
from src.utils.common import read_yaml
from src.utils.common import create_directories
from src.constants import CONFIG_FILE_PATH
from src.constants import SCHEMA_FILE_PATH
from src.config.configuration_data_class import DataValidationConfig
from src.config.configuration_data_class import DataSplittingConfig
from src.config.configuration_data_class import BuildFeaturesConfig
from src.config.configuration_data_class import TrainingConfig
from src.config.configuration_data_class import PredictionConfig


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
        self.schema = read_yaml(Path(schema_filepath))

    @property
    def data_validation_config(self) -> DataValidationConfig:
        """
        Get configuration for data validation.

        Returns:
            DataValidationConfig: Configuration for data validation.
        """
        config = self.config.data_validation
        schema = self.schema
        data_validation_config = DataValidationConfig(
            raw_data_file=config.raw_data_file,
            schema=schema,
        )
        return data_validation_config

    @property
    def data_splitting_config(self) -> DataSplittingConfig:
        """
        Get configuration for data splitting.

        Returns:
            DataSplittingConfig: Configuration for data splitting.
        """
        config = self.config.data_splitting
        data_splitting_config = DataSplittingConfig(
            raw_data_file=config.raw_data_file,
            train_file=config.train_file,
            test_file=config.test_file,
            target=config.target,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        return data_splitting_config

    @property
    def build_features_config(self) -> BuildFeaturesConfig:
        """Get configuration for building features.

        Returns:
            BuildFeaturesConfig: Configuration for building features.
        """
        config = self.config.build_features
        create_directories([config.artifacts_dir])

        build_features_config = BuildFeaturesConfig(
            train_file=config.train_file,
            target=config.target,
            processed_train_file=config.processed_train_file,
            transformer_file=config.transformer_file,
        )
        return build_features_config

    @property
    def model_training_config(self) -> TrainingConfig:
        """
        Get configuration for model training.

        Returns:
            ModelTrainingConfig: Configuration for model training.
        """
        config = self.config.model_training
        params = self.params
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_training_config = TrainingConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            experiment_name=config.mlflow.experiment_name,
            registered_model_name=config.mlflow.registered_model_name,
            target_column=schema.name,
            binning_process=params.binning_process,
            logistic_regression=params.logistic_regression,
            scorecard=params.scorecard,
        )
        return model_training_config

    @property
    def model_inference_config(self) -> PredictionConfig:
        """
        Get configuration for model inference.

        Returns:
            PredictionConfig: Configuration for model inference.
        """
        config = self.config.model_inference

        create_directories([config.root_dir])

        model_inference_config = PredictionConfig(
            registered_model_name=config.mlflow.registered_model_name,
            version=config.mlflow.version,
        )
        return model_inference_config
