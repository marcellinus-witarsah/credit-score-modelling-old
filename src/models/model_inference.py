import joblib
import numpy as np
import pandas as pd
import mlflow
import os
from typing import Union
from src.utils.common import logger
from sklearn.base import BaseEstimator
from src.entities.config_entity import ModelInferenceConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class ModelInference:
    """
    A class used to perform model inference using a pre-trained model.

    This class is responsible for loading a model from a specified path and
    providing methods to make predictions on input data.

    Attributes:
        config (ModelInferenceConfig): Configuration for model inference.
        model: The loaded machine learning model.
    """

    def __init__(self, config: ModelInferenceConfig):
        """
        Initialize the ModelInference with a configuration.

        Args:
            config (ModelInferenceConfig): The configuration containing paths for model inference.
        """
        self.config = config
        self.model = self.get_model(
            self.config.registered_model_name, self.config.version
        )

    def get_model(self, model_name: str, version: int) -> BaseEstimator:
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
            return model
        except Exception as e:
            logger.error(e)

    def predict(self, data: Union[pd.DataFrame, np.ndarray, np.array]) -> np.array:
        """
        Make predictions on input data.

        Args:
            data Union[pd.DataFrame, np.ndarray, np.array]: Preprocessed input data for which predictions are to be made.

        Returns:
            np.array: The predicted values.
        """
        logger.info("Predict")
        prediction = self.model.predict(data)
        return prediction

    def predict_proba(
        self, data: Union[pd.DataFrame, np.ndarray, np.array]
    ) -> np.array:
        """
        Make probability predictions on input data.

        Args:
            data Union[pd.DataFrame, np.ndarray, np.array]: Preprocessed input data for which probability predictions are to be made.

        Returns:
            np.array: The predicted probabilities.
        """
        logger.info("Predict probabilities")
        prediction = self.model.predict_proba(data)
        return prediction[:, -1]

    def score(self, data: Union[pd.DataFrame, np.ndarray, np.array]) -> np.array:
        """
        Give credit scores on input data.

        Args:
            data Union[pd.DataFrame, np.ndarray, np.array]: Preprocessed input data for which credit scores are to be made.

        Returns:
            np.array: The credit scores.
        """
        logger.info("Get credit score")
        prediction = self.model.score(data)
        return prediction
