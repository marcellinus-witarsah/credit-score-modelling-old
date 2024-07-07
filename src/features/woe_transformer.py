import numpy as np
import pandas as pd
from typing import Union


class WOETransformer:
    def __init__(self):
        self.__woe_dict = {}
        self.__woe_df = pd.DataFrame()
        self.__iv_df = pd.DataFrame()

    @property
    def iv_df(self) -> pd.DataFrame:
        return self.__iv_df.copy()

    @property
    def woe_df(self) -> pd.DataFrame:
        return self.__woe_df.copy()

    @property
    def woe_dict(self) -> dict:
        return self.__woe_dict.copy()

    def __interpret_information_value(self, value: float) -> str:
        """Interpret the strength of the informaiton value.

        Args:
            value (float): information value.

        Returns:
            str: interpretation of the information value.
        """
        if value < 0.02:
            return "Not Predictive"
        elif 0.02 <= value and value < 0.1:
            return "Weak Predictive"
        elif 0.1 <= value and value < 0.3:
            return "Medium Predictive"
        elif 0.3 <= value and value < 0.5:
            return "Strong Predictive"
        else:
            return "Very Strong Predictive"

    def __get_woe(self, value: Union[int, float], feature_name: str) -> float:
        """Tranform column values into WOE values.

        Args:
            value (Union[int, float]): column value to be mapped to WOE value.

        Returns:
            float: WOE value.
        """
        woe_value = None
        if pd.isna(value):
            woe_value = self.__woe_dict[feature_name].get("Missing", None)
        else:
            for interval, woe in self.__woe_dict[feature_name].items():
                if isinstance(interval, pd.Interval) and value in interval:
                    woe_value = woe
        return woe_value

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Generate the weight of evidence values based on its features and target.

        Args:
            features (pd.DataFrame): features for the prediction.
            target (pd.Series): target prediction.
        """
        # 1. Count the occurences of the target inside each bin of the features.
        crosstabs = {}
        for col in features.columns:
            crosstabs[col] = pd.crosstab(features[col], target, margins=True)

        # 2. Calculate weight of evidence for all features
        for col, crosstab in crosstabs.items():
            # 2.1 Calculate WOE
            crosstab["proportion_not_default"] = crosstab[0] / crosstab[0]["All"]
            crosstab["proportion_default"] = crosstab[1] / crosstab[1]["All"]
            crosstab["WOE"] = np.log(
                crosstab["proportion_not_default"] / crosstab["proportion_default"]
            )
            temp_df = crosstab.reset_index().iloc[:-1, [0, -1]].copy()
            temp_df.columns = ["Value", "WOE"]
            # self.__woe_dict[col] = temp_df.set_index("Value")["WOE"].to_dict()

            self.__woe_dict[col] = {
                row["Value"]: row["WOE"] for _, row in temp_df.iterrows()
            }  # for woe values mapping

            temp_df.loc[:, "Characteristic"] = col
            self.__woe_df = pd.concat(
                (self.__woe_df, temp_df), axis=0
            )  # for pandas dataframe

            # 2.2 Calculate information value
            IV = np.sum(
                (crosstab["proportion_not_default"] - crosstab["proportion_default"])
                * crosstab["WOE"]
            )
            self.__iv_df = pd.concat(
                (
                    self.__iv_df,
                    pd.DataFrame({"Characteristic": [col], "Information Value": [IV]}),
                ),
                axis=0,
            )
            self.__iv_df["Interpretation"] = self.__iv_df["Information Value"].apply(
                lambda x: self.__interpret_information_value(x)
            )
            self.__iv_df = self.__iv_df.sort_values(
                by="Information Value", ascending=False
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input data into weight of evidence values.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Output dataframe.
        """
        # 1. Map numerical features with their weight of evidence values
        for col in df.select_dtypes("number").columns:
            df[col] = df[col].apply(lambda x: self.__get_woe(x, col))
            df[col] = df[col].fillna(self.__woe_dict[col].get("Missing", np.nan))

        # 2. Map categorical features with their weight of evidence values
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].map(self.__woe_dict[col])
            df[col] = df[col].fillna(self.__woe_dict[col].get("Missing", np.nan))
        return df
