import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.utils.common import logger
from src.models.model_strategy import ModelStrategy
from src.models.metrics import roc_auc, pr_auc, gini, ks


class LogisticRegressionModel(ModelStrategy):

    def __init__(self, model):
        self.__model = model
        logger.info("Instantiated {} model".format(self.__model.__class__.__name__))

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.info("Load {} model from {} file".format(model.__class__.__name__, file))
        return cls(model)

    @classmethod
    def from_parameters(cls, parameters):
        model = LogisticRegression(**parameters)
        logger.info("{} model created".format(model.__class__.__name__))
        return cls(model)

    @property
    def model(self):
        return self.__model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        start_time = time.perf_counter()
        self.__model.fit(X_train, y_train)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model training finished in {:.2f} seconds.".format(
                self.__model.__class__.__name__, elapsed_time
            )
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        start_time = time.perf_counter()
        result = self.__model.predict(X_test)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model prediction finished in {:.2f} seconds.".format(
                self.__model.__class__.__name__, elapsed_time
            )
        )
        return result

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        start_time = time.perf_counter()
        result = self.__model.predict_proba(X_test)[:, 1]
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model prediction probability finished in {:.2f} seconds.".format(
                self.__model.__class__.__name__, elapsed_time
            )
        )
        return result

    def evaluate(
        self, X_test: pd.DataFrame, y_true: pd.Series, type: str = "Training"
    ) -> None:
        start_time = time.perf_counter()
        y_pred = self.predict_proba(X_test)
        logger.info(
            "{} Performance >>> ROC AUC: {:.2f}, PR AUC: {:.2f}, GINI: {:.2f}, KS SCORE: {:.2f}".format(
                type,
                roc_auc(y_true, y_pred),
                pr_auc(y_true, y_pred),
                gini(y_true, y_pred),
                ks(y_true, y_pred),
            )
        )
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model evaluation finished in {:.2f} seconds.".format(
                self.__model.__class__.__name__, elapsed_time
            )
        )

    def save(self, file: Path) -> None:
        start_time = time.perf_counter()
        with open(file, "wb") as file:
            pickle.dump(self.__model, file)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Save {} model finished in {:.2f} seconds.".format(
                self.__model.__class__.__name__, elapsed_time
            )
        )
