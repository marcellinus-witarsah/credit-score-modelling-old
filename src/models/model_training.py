import pandas as pd
from sklearn.linear_model import LogisticRegression
from optbinning import Scorecard
from optbinning import BinningProcess
from src.utils.common import logger
from src.entities.config_entity import ModelTrainingConfig


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
        Train and save the model.
        """
        logger.info("Train model")
        train = pd.read_csv(self.config.train_data_path)

        X_train = train.drop(columns=[self.config.target_column])
        y_train = train[self.config.target_column]

        # Instantiate BinningProcess
        binning_process = BinningProcess(
            X_train.columns.values, **self.config.BinningProcess
        )
        # Instantiate LogisticRegression
        logreg_model = LogisticRegression(**self.config.LogisticRegression)

        # Instantiate Scorecard Model
        scorecard = Scorecard(
            binning_process=binning_process,
            estimator=logreg_model,
            **self.config.Scorecard
        )

        # Train
        scorecard.fit(X_train, y_train)

        # Save model
        scorecard.save(self.config.model_path)
