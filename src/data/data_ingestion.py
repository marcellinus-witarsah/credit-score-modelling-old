import polars as pl
import pandas as pd
from pathlib import Path
from abc import ABC
from abc import abstractmethod
from src.utils.common import logger
from src.utils.common import read_yaml
from src.utils.common import create_directories
from src.entities.config_entity import DataIngestionConfig


class DataIngestionStrategy(ABC):
    """
    Abstract base class for data ingestion strategies.
    """

    @abstractmethod
    def ingest_data(self, paths: list, target_path: Path) -> None:
        """
        Ingests data from the specified paths to the target path.

        Args:
            paths (list): List of file paths to ingest data from.
            target_path (Path): Path to save the ingested data.
        """
        pass


class PandasDataIngestionStrategy(DataIngestionStrategy):
    """
    Data ingestion strategy using Pandas.
    """

    def ingest_data(self, paths: list, target_path: Path) -> None:
        """
        Ingests data using Pandas.

        Args:
            paths (list): List of file paths to ingest data from.
            target_path (Path): Path to save the ingested data.

        Returns:
            None
        """
        df = pd.DataFrame({})

        # Read and concatenate dataframes (if more than one):
        for path in paths:
            path = Path(path)
            if path.suffix == ".csv":
                temp_df = pd.read_csv(path)
            elif path.suffix == ".parquet":
                temp_df = pd.read_parquet(path)
            df = pd.concat([df, temp_df], axis=0)

        # Save data to target_path:
        df.to_csv(target_path, index=False)
        logger.info(f"Data saved to {target_path}")


class PolarsDataIngestionStrategy(DataIngestionStrategy):
    """
    Data ingestion strategy using Polars.
    """

    def ingest_data(self, paths: list, target_path: Path) -> None:
        """
        Ingests data using Polars.

        Args:
            paths (list): List of file paths to ingest data from.
            target_path (Path): Path to save the ingested data.

        Returns:
            None
        """
        df = None

        # Read and concatenate lazyframes (if more than one):
        for path in paths:
            path = Path(path)
            if path.suffix == ".csv":
                temp_df = pl.scan_csv(path)
            elif path.suffix == ".parquet":
                temp_df = pl.scan_parquet(path)
            if df is None:
                df = temp_df
            else:
                df = pl.concat([df, temp_df], how="vertical")

        # Save data to target_path:
        df.sink_csv(target_path, index=False)
        logger.info(f"Data saved to {target_path}")


class DataIngestion:
    """
    Class to manage data ingestion process.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Instantiate `DataIngestion` class.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config

    def ingest_data(self, strategy: DataIngestionStrategy) -> None:
        """
        Ingests data using the specified strategy.

        Args:
            strategy (DataIngestionStrategy): Strategy to use for data ingestion.

        Returns:
            None
        """
        try:
            # Check if the path is a string or list of paths:
            paths = (
                self.config.source_path
                if isinstance(self.config.source_path, list)
                else [self.config.source_path]
            )
            strategy.ingest_data(paths, self.config.target_path)
            logger.info(
                f"Successfully ingested data using {strategy.__class__.__name__}"
            )
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
