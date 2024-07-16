import pandas as pd
from src.models.woe_logistic_regression import WOELogisticRegression
from src.config.configuration_manager import ConfigurationManager


def predict():
    evaluate = ConfigurationManager().evaluate_config

    # 1. Load data
    test_df = pd.read_csv(evaluate.test_file)
    X_test, y_test = (
        test_df.drop(columns=[evaluate.target]),
        test_df[evaluate.target],
    )

    # 2. Initialize model
    model = WOELogisticRegression.from_file(evaluate.model_file)

    # 3. Evaluate testing performance
    model.evaluate(X_test, y_test, "Testing")


if __name__ == "__main__":
    predict()
