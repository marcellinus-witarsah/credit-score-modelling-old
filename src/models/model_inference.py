import joblib
import numpy as np
import pandas as pd
from typing import Union
from src.utils.common import logger
from src.entities.config_entity import ModelInferenceConfig


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
        self.model = self.get_model()

    def get_model(self):
        """
        Load the model from the file specified in the configuration.

        This method reads the model file from the path specified in the config
        and loads it into memory.

        Returns:
            model: The loaded machine learning model.
        """
        logger.info("Load model")
        model = None
        with open(self.config.model_path, "rb") as f:
            model = joblib.load(f)
        return model

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
