from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Data class for storing data validation configuration.

    Attributes:
        root_dir (Path): Root directory for data validation.
        source_path (Path): Source path of the data to be validated.
        STATUS_FILE (Path): Path to the status file.
        schema (list): List defining the schema for validation.
    """

    raw_data_file: Path
    schema: list


@dataclass(frozen=True)
class DataSplittingConfig:
    """
    Data class for storing data splitting configuration.

    Attributes:
        root_dir (Path): Root directory for data splitting.
        source_path (Path): Source path of the data to be processed.
        train_data_path (Path): Path to save the training data.
        test_data_path (Path): Path to save the testing data.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        random_state (int): Random seed for reproducibility.
    """

    raw_data_file: Path
    train_file: Path
    test_file: Path
    target: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class BuildFeaturesConfig:
    """
    Data class for storing data splitting configuration.

    Attributes:
        root_dir (Path): Root directory for data splitting.
        source_path (Path): Source path of the data to be processed.
        train_data_path (Path): Path to save the training data.
        test_data_path (Path): Path to save the testing data.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        random_state (int): Random seed for reproducibility.
    """

    train_file: Path
    target: str
    processed_train_file: Path
    transformer_file: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class TrainConfig:
    """
    Data class for storing model training configuration.

    Attributes:
        root_dir (str): Root directory for model inference.
        model_path (str): Path to save the trained model.
        train_data_path (str): Path to the inference data.
        test_data_path (str): Path to the test data.
        experiment_name (str): Name of the experiment.
        registered_model_name (str): Model name.
        target_column (str): The name of the target column.
        binning_process (dict): Configuration for the binning process.
        logistic_regression (dict): Configuration for logistic regression.
        scorecard (dict): Configuration for the scorecard.
    """

    processed_train_file: Path
    woe_transformer_params: dict
    logreg_params: dict
    artifacts_dir: Path
    model_file: Path
    transformer_file: Path
    target: str
    test_file: Path


@dataclass(frozen=True)
class EvaluateConfig:
    """
    Data class for storing model evaluation configuration.

    Attributes:
        model_file (Path): Path to save the trained model.
        test_file (Path): Path to the test data.
        target (str): The name of the target column.
    """

    test_file: Path
    model_file: Path
    target: str


@dataclass(frozen=True)
class PredictionConfig:
    """
    Data class for storing model prediction configuration.

    Attributes:
        registered_model_name (str): The name of the registered model.
        version (int): The version number of the model.
    """

    artifacts_dir: Path
    model_file: Path
    transformer_file: Path
    target: str
    test_file: Path
