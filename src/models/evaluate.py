import pandas as pd
import pickle
from src.models.woe_logistic_regression import WOELogisticRegression
from src.config.configuration_manager import ConfigurationManager


def predict():
    prediction_config = ConfigurationManager().prediction_config

    # 1. Load data
    test_df = pd.read_csv(prediction_config.test_file)
    X_test, y_test = (
        test_df.drop(columns=[prediction_config.target]),
        test_df[prediction_config.target],
    )

    # 2. Initialize model
    model = WOELogisticRegression.from_file(prediction_config.model_file)

    # 3. Evaluate testing performance
    model.evaluate(X_test, y_test, "Testing")


if __name__ == "__main__":
    predict()
