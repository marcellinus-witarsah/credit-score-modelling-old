import mlflow
import pandas as pd
import os
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
from mlflow.models import infer_signature
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegression
from optbinning import Scorecard
from optbinning import BinningProcess
from src.utils.common import logger
from src.visualization.visualize import plot_calibration_curve
from src.entities.config_entity import ModelTrainingConfig
from src.models.metrics import roc_auc, pr_auc, gini, ks

# Load environment varible:
load_dotenv(find_dotenv())


from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple
from dotenv import load_dotenv, find_dotenv
import mlflow
from urllib.parse import urlparse
from mlflow.models import infer_signature

load_dotenv(find_dotenv())


class ModelTraining:
    """
    Class to handle the model training process.
    """

    def __init__(self, config: ModelTrainingConfig):
        """
        Instantiate `ModelTraining` class.

        Args:
            config (ModelTrainingConfig): Configuration for model training.
        """
        self.config = config

    def train(self) -> None:
        """
        Train, evaluate, and log model to MLFlow Registry.
        """
        logger.info("Train model")
        train = pd.read_csv(self.config.train_data_path)
        X_train = train.drop(columns=[self.config.target_column])
        y_train = train[self.config.target_column]

        # Instantiate BinningProcess:
        binning_process = BinningProcess(
            X_train.columns.values, **self.config.binning_process
        )

        # Instantiate LogisticRegression:
        logreg_model = LogisticRegression(**self.config.logistic_regression)

        # Instantiate Scorecard:
        scorecard_model = Scorecard(
            binning_process=binning_process,
            estimator=logreg_model,
            **self.config.scorecard
        )

        # Train:
        scorecard_model.fit(X_train, y_train)

        # Predictin on Train Data:
        y_pred_proba_train = scorecard_model.predict_proba(X_train)[:, -1]

        # Predictin on Test Data:
        test = pd.read_csv(self.config.test_data_path)
        X_test = test.drop(columns=[self.config.target_column])
        y_test = test[self.config.target_column]
        y_pred_proba_test = scorecard_model.predict_proba(X_test)[:, -1]

        # Track Experiment using MLFlow:
        logger.info("Initialize MLFlow Tracking ...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRAKING_URI"))
        mlflow.set_experiment(self.config.experiment_name)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        logger.info("Start Tracking ...")
        # Track Experiment:
        with mlflow.start_run():
            logger.info("Log Params")
            mlflow.log_params(self.config.binning_process)
            mlflow.log_params(self.config.logistic_regression)
            mlflow.log_params(self.config.scorecard)

            # Log Train Metrics
            logger.info("Log Metrics")
            mlflow.log_metric("train_roc_auc", roc_auc(y_train, y_pred_proba_train))
            mlflow.log_metric("train_pr_auc", pr_auc(y_train, y_pred_proba_train))
            mlflow.log_metric("train_gini", gini(y_train, y_pred_proba_train))
            mlflow.log_metric("train_ks", ks(y_train, y_pred_proba_train))

            # log Test Metrics:
            mlflow.log_metric("test_roc_auc", roc_auc(y_test, y_pred_proba_test))
            mlflow.log_metric("test_pr_auc", pr_auc(y_test, y_pred_proba_test))
            mlflow.log_metric("test_gini", gini(y_test, y_pred_proba_test))
            mlflow.log_metric("test_ks", ks(y_test, y_pred_proba_test))

            # Log Models:
            logger.info("Log Models")
            signature = infer_signature(X_train.iloc[:1, :], y_pred_proba_train[:1])
            mlflow.sklearn.log_model(
                scorecard_model,
                "model",
                signature=signature,
                registered_model_name=self.config.registered_model_name,
            )

            # Log Plots:
            logger.info("Log Artifacts")
            image = plot_calibration_curve(
                y_train,
                y_pred_proba_train,
                "Logistic Regression",
                (10, 7),
                self.config.root_dir + "/train_model_calibration.png",
            )
            mlflow.log_artifact(self.config.root_dir + "/train_model_calibration.png")
            image = plot_calibration_curve(
                y_test,
                y_pred_proba_test,
                "Logistic Regression",
                (10, 7),
                self.config.root_dir + "/test_model_calibration.png",
            )
            mlflow.log_artifact(self.config.root_dir + "/test_model_calibration.png")
