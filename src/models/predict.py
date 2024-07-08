import pandas as pd
import pickle
from src.models.logistic_regression import LogisticRegressionModel
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
    model = LogisticRegressionModel.from_file(prediction_config.model_file)

    # 3. Evaluate testing performance
    with open(prediction_config.transformer_file, "rb") as f:
        woe_transformer = pickle.load(f)
    model.evaluate(woe_transformer.transform(X_test), y_test, "Testing")


if __name__ == "__main__":
    predict()
