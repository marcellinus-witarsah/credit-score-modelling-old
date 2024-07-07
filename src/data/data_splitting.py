import pandas as pd
import time
from typing import Tuple
from src.utils.common import logger
from sklearn.model_selection import train_test_split
from src.config.configuration_manager import ConfigurationManager


# src/data/data_preprocessing.py
class DataSplitting:
    """
    Class to handle the data preprocessing process.
    """

    def __init__(self):
        """
        Instantiate `DataPreprocessing` class.

        Args:
            config (DataPreprocessingConfig): Configuration for data preprocessing.
        """
        self.config = ConfigurationManager().data_preprocessing_config

    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train and test data evenly based on their target values.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train and test set.
        """
        try:
            start_time = time.perf_counter()
            logger.info("Split data")

            # 1. Load data:
            df = pd.read_csv(self.config.raw_data_file)

            # 2. Separate between features and label
            X, y = (
                df.drop(columns=[self.config.target]),
                df[self.config.target],
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
            train.to_csv(self.config.train_file, index=False)
            test.to_csv(self.config.test_file, index=False)

            elapsed_time = time.perf_counter() - start_time
            logger.info("Split data finished in {:.2f} seconds.".format(elapsed_time))

        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    data_preprocessing.run()
