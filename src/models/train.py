import pandas as pd
import pickle
from src.models.logistic_regression import LogisticRegressionModel
from src.config.configuration_manager import ConfigurationManager


def train():
    train_config = ConfigurationManager().train_config

    # 1. Load data
    train_df = pd.read_csv(train_config.processed_train_file)
    test_df = pd.read_csv(train_config.test_file)
    X_train, y_train = (
        train_df.drop(columns=[train_config.target]),
        train_df[train_config.target],
    )

    X_test, y_test = (
        test_df.drop(columns=[train_config.target]),
        test_df[train_config.target],
    )

    # 2. Initialize model
    model = LogisticRegressionModel(train_config.model_params)

    # 3. Train model
    model.fit(X_train, y_train)

    # 4. Evaluate training performance
    model.evaluate(X_train, y_train, "")

    # 5. Evaluate testing performance
    with open(train_config.transformer_file, "rb") as f:
        woe_transformer = pickle.load(f)
    model.evaluate(woe_transformer.transform(X_test), y_test, "Testing")

    # 6. Save model
    model


if __name__ == "__main__":
    train()
