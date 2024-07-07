# -*- coding: utf-8 -*-
import time
from src.data.data_splitting import DataSplitting
from src.data.data_validation import DataValidation
from src.utils.common import logger


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    data_validation = DataValidation()
    data_validation.run()

    data_preprocessing = DataSplitting()
    data_preprocessing.run()


if __name__ == "__main__":
    main()
