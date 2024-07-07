import pickle
import pandas as pd
from src.utils.common import logger
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

    # 2. Generate bins
    X_train_binned = generate_bins(
        X_train.copy(), X_train.select_dtypes("number").columns.tolist(), 5
    )

    # 3. Fill missing categorical values
    X_train_binned = fill_missing_categorical(X_train_binned)

    # 4. Transform into weight of evidence values
    woe_transformer = WOETransformer()
    woe_transformer.fit(X_train_binned, y_train)
    X_train = woe_transformer.transform(X_train)

    # 5. Save the built features for training and the transformer object
    pd.concat([X_train, y_train], axis=1).to_csv(
        build_features_config.processed_train_file
    )
    with open(build_features_config.transformer_file, "wb") as file:
        pickle.dump(woe_transformer, file)


if __name__ == "__main__":
    main()
