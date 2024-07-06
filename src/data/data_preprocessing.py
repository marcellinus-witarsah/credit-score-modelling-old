import pandas as pd
import time
from typing import Tuple
from src.utils.common import logger
from sklearn.model_selection import train_test_split
from src.entities.config_entity import DataPreprocessingConfig


# src/data/data_preprocessing.py
class DataPreprocessing:
    """
    Class to handle the data preprocessing process.
    """

    def __init__(self):
        """
        Instantiate `DataPreprocessing` class.

        Args:
            config (DataPreprocessingConfig): Configuration for data preprocessing.
        """
        self.config = DataPreprocessingConfig

    def split(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train and test data evenly based on their target values.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train and test set.
        """
        try:
            start_time = time.perf_counter()
            logger.info("Split data")

            # 1. Load data:
            df = pd.read_csv(self.config.source_path)

            # 2. Separate between features and label
            X, y = (
                df.drop(columns=[self.config.target_column]),
                df[self.config.target_column],
            )

            # 3. Split Data:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                stratify=y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            # 4. Concat into a DataFrame:
            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            # 5. Save data:
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)

            elapsed_time = time.perf_counter() - start_time
            logger.info("Finished in {:.2f} seconds.".format(elapsed_time))

        except Exception as e:
            logger.error(e)
