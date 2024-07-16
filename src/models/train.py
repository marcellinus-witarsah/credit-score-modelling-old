import pandas as pd
from pathlib import Path
from src.models.woe_logistic_regression import WOELogisticRegression
from src.config.configuration_manager import ConfigurationManager
from src.utils.common import save_json
from src.visualization.visualize import plot_calibration_curve


def train():
    train_config = ConfigurationManager().train_config

    # 1. Load data
    train_df = pd.read_csv(train_config.train_file)
    test_df = pd.read_csv(train_config.test_file)
    X_train, y_train = (
        train_df.drop(columns=[train_config.target]),
        train_df[train_config.target],
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
    plot_calibration_curve(
        y_true=y_train,
        y_pred_proba=model.predict_proba(X_train)[:, 1],
        model_name=model.__class__.__name__,
        path=train_config.calibration_curve_file,
    )


if __name__ == "__main__":
    train()
