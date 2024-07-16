import pandas as pd
from src.models.woe_logistic_regression import WOELogisticRegression
from src.config.configuration_manager import ConfigurationManager
from src.visualization.visualize import plot_calibration_curve


def predict():
    evaluate_config = ConfigurationManager().evaluate_config

    # 1. Load data
    test_df = pd.read_csv(evaluate_config.test_file)
    X_test, y_test = (
        test_df.drop(columns=[evaluate_config.target]),
        test_df[evaluate_config.target],
    )

    # 2. Initialize model
    model = WOELogisticRegression.from_file(evaluate_config.model_file)

    # 3. Evaluate testing performance
    model.evaluate(X_test, y_test, "Testing")
    plot_calibration_curve(
        y_true=y_test,
        y_pred_proba=model.predict_proba(X_test)[:, 1],
        model_name=model.__class__.__name__,
        path=evaluate_config.calibration_curve_file,
    )


if __name__ == "__main__":
    predict()
