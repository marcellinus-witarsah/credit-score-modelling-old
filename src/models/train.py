import pandas as pd
from pathlib import Path
from src.models.woe_logistic_regression import WOELogisticRegression
from src.config.configuration_manager import ConfigurationManager
from src.utils.common import load_pickle, save_json, save_pickle


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
    model = WOELogisticRegression.from_parameters(
        woe_transformer_params=train_config.woe_transformer_params,
        logreg_params=train_config.logreg_params,
    )

    # 3. Train model
    model.fit(X_train, y_train)

    # 4. Evaluate training performance
    roc_auc_score, pr_auc_score, gini_score, ks_score = model.evaluate(
        X_train, y_train, "Train"
    )
    save_json(
        Path("reports/train_evaluation_metric.json"),
        {
            "roc_auc_score": roc_auc_score,
            "pr_auc_score": pr_auc_score,
            "gini_score": gini_score,
            "ks_score": ks_score,
        },
    )

    # 5. Evaluate testing performance
    model.evaluate(X_test, y_test, "Testing")
    save_json(
        Path("reports/test_evaluation_metric.json"),
        {
            "roc_auc_score": roc_auc_score,
            "pr_auc_score": pr_auc_score,
            "gini_score": gini_score,
            "ks_score": ks_score,
        },
    )

    # 6. Save model
    model.save(file=train_config.model_file)


if __name__ == "__main__":
    train()
