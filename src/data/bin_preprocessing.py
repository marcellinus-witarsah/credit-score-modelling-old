import pandas as pd
import time
from optbinning import BinningProcess
from src.utils.common import logger


class BinPreprocessing:
    __binning_process = None

    def __init__(self, variables: list, categorical_variables: list):
        """Instantiate `BinProcessing` class

        Args:
            variables (list): list of variable names to be binned
            categorical_variables (list): list of categorical variable names
        """
        self.__binning_process = BinningProcess(
            variables, categorical_variables=categorical_variables
        )

    def fit(self, variables: pd.DataFrame, target: pd.Series):
        """Perform binning

        Args:
            variables (pd.DataFrame): variables to be binned
            target (pd.Series): target variable
        """
        try:
            logger.info("Fit data for binning ...")
            start_time = time.perf_counter()
            self.__binning_process.fit(variables, target)
            elapsed_time = time.perf_counter() - start_time
            logger.info("DONE in {:.3f} s ".format(elapsed_time))
        except Exception as e:
            logger.error("FAILED: {}".format(e))

    @property
    def binning_process(self) -> BinningProcess:
        return self.__binning_process
