from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class for storing data ingestion configuration.

    Attributes:
        root_dir (Path): Root directory for data ingestion.
        source_path (Path): Source path of the data.
        target_path (Path): Target path for the processed data.
    """

    root_dir: Path
    source_path: Path
    target_path: Path


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

    root_dir: Path
    source_path: Path
    STATUS_FILE: Path
    schema: list


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """
    Data class for storing data preprocessing configuration.

    Attributes:
        root_dir (Path): Root directory for data preprocessing.
        source_path (Path): Source path of the data to be processed.
        train_data_path (Path): Path to save the training data.
        test_data_path (Path): Path to save the testing data.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        random_state (int): Random seed for reproducibility.
    """

    root_dir: Path
    source_path: Path
    train_data_path: Path
    test_data_path: Path
    target_column: str
    test_size: float
    shuffle: bool
    random_state: int
