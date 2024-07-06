import pandas as pd
import time
from typing import Tuple
from src.utils.common import logger
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame, target: str, test_size: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train and test data evenly based on their target values.

    Args:
        df (pd.DataFrame): _description_
        target (str): _description_
        test_size (float): _description_
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train and test set.
    """
    try:
        logger.info("Split data for training and testing ...")
        start_time = time.perf_counter()
        X, y = df.drop(columns=[target]), df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )
        elapsed_time = time.perf_counter() - start_time
        logger.info("DONE in {:.3f} s ".format(elapsed_time))
        return (X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.error("FAILED: {}".format(e))
