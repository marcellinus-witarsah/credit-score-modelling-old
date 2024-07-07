import time
import pandas as pd
from src.utils.common import logger


def generate_bins(
    df: pd.DataFrame, numerical_columns: str, num_of_bins: int
) -> pd.DataFrame:
    """
    Create bins for a numerical column, dividing it into a specified number of equal-sized bins.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the data.
        numerical_column (str): Numerical column.
        num_of_bins (int): Number of bins to create.
    Returns:
        pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
    """

    try:
        start_time = time.perf_counter()
        logger.info("Generate bins")

        for numerical_column in numerical_columns:
            df[numerical_column] = pd.qcut(
                df[numerical_column], q=num_of_bins, duplicates="drop"
            )
        elapsed_time = time.perf_counter() - start_time

        logger.info("Generate bins finished in {:.2f} seconds.".format(elapsed_time))
        return df
    except Exception as e:
        logger.error(e)


def fill_missing_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing categorical columns inside Pandas DataFrame with `Missing`.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the data.
    Returns:
        pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
    """
    try:
        start_time = time.perf_counter()
        logger.info("Fill missing categories")

        for column in df.columns:
            if df[column].isna().sum() > 0 and df[column].dtype in [
                "object",
                "category",
            ]:
                # Add category 'Missing' to replace the missing values
                df[column] = df[column].cat.add_categories("Missing")
                # Replace missing values with category 'Missing'
                df[column] = df[column].fillna(value="Missing")

        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Fill missing categories finished in {:.2f} seconds.".format(elapsed_time)
        )
        return df
    except Exception as e:
        logger.error(e)
