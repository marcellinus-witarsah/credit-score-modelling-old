from pathlib import Path
from src.utils.common import read_yaml
from src.utils.common import create_directories
from src.constants import FileConstants
from src.config.configuration_data_class import DataValidationConfig
from src.config.configuration_data_class import DataSplittingConfig
from src.config.configuration_data_class import BuildFeaturesConfig
from src.config.configuration_data_class import TrainConfig
from src.config.configuration_data_class import EvaluateConfig
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
        config_filepath: str = FileConstants.CONFIG_FILE_PATH.value,
        schema_filepath: str = FileConstants.SCHEMA_FILE_PATH.value,
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
            artifacts_dir=config.artifacts_dir,
        )
        return build_features_config

    @property
    def train_config(self) -> TrainConfig:
        """
        Get configuration for model training.

        Returns:
            TrainConfig: Configuration for model training.
        """
        config = self.config.train

        create_directories([config.artifacts_dir])

        train_config = TrainConfig(
            processed_train_file=config.processed_train_file,
            woe_transformer_params=config.woe_transformer_params,
            logreg_params=config.logreg_params,
            artifacts_dir=config.artifacts_dir,
            model_file=config.model_file,
            transformer_file=config.transformer_file,
            target=config.target,
            test_file=config.test_file,
        )
        return train_config

    @property
    def evaluate_config(self) -> EvaluateConfig:
        """
        Get configuration for model evaluation.

        Returns:
            EvaluateConfig: Configuration for model evaluation.
        """
        config = self.config.evaluate
        evaluate_config = EvaluateConfig(
            test_file=config.test_file,
            model_file=config.model_file,
            target=config.target,
        )
        return evaluate_config

    @property
    def prediction_config(self) -> PredictionConfig:
        """
        Get configuration for model inference.

        Returns:
            PredictionConfig: Configuration for model inference.
        """
        config = self.config.predict

        prediction_config = PredictionConfig(
            artifacts_dir=config.artifacts_dir,
            model_file=config.model_file,
            transformer_file=config.transformer_file,
            target=config.target,
            test_file=config.test_file,
        )
        return prediction_config
