import pandas as pd
from src.utils.common import logger
from src.utils.common import save_pickle
from src.config.configuration_manager import ConfigurationManager
from src.features.utility import generate_bins
from src.features.utility import fill_missing_categorical
from src.features.woe_transformer import WOETransformer


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    build_features_config = ConfigurationManager().build_features_config

    # 1. Load training data
    train_df = pd.read_csv(build_features_config.train_file)
    X_train, y_train = (
        train_df.drop(columns=[build_features_config.target]),
        train_df[build_features_config.target],
    )

    # 2. Transform into weight of evidence values
    woe_transformer = WOETransformer(
        numerical_features=X_train.select_dtypes("number").columns,
        categorical_features=X_train.select_dtypes(["object", "category"]).columns,
        bins=4,
    )
    woe_transformer = woe_transformer.fit(X_train, y_train)
    X_train = woe_transformer.transform(X_train)

    # 3. Save the built features for training and the transformer object
    pd.concat([X_train, y_train], axis=1).to_csv(
        build_features_config.processed_train_file, index=False
    )
    save_pickle(
        data=woe_transformer, path=build_features_config.transformer_file, mode="wb"
    )


if __name__ == "__main__":
    main()
