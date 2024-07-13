import pandas as pd
import numpy as np
from typing import Union
from sklearn.linear_model import LogisticRegression
from src.features.woe_transformer import WOETransformer


class CreditScoreScaling:
    def __init__(
        self,
        pipeline,
        pdo: Union[int, float],
        odds: Union[int, float],
        scorecard_points: Union[int, float],
    ):
        self.pipeline = pipeline
        self.intercept = self.pipeline[LogisticRegression.__name__].intercept_[0]
        self.n_features = len(
            self.pipeline[LogisticRegression.__name__].feature_names_in_
        )
        self.pdo = pdo
        self.odds = odds
        self.scorecard_points = scorecard_points
        self.factor = self.pdo / np.log(2)
        self.offset = self.scorecard_points - self.factor * np.log(self.odds)
        self.scorecard_map_dict = {}  # for mapping values to its credit point

    # Private methods
    def __generate_mapping(self, scorecard: pd.DataFrame) -> None:
        # Create dictionary mapping in the background
        scorecard_features = scorecard["Characteristic"].unique()

        for feature in scorecard_features:
            self.scorecard_map_dict[feature] = {
                row["Value"]: row["Points"]
                for _, row in scorecard[
                    scorecard["Characteristic"] == feature
                ].iterrows()
            }

    def __get_credit_point(self, feature: str, value: Union[int, float, str]) -> float:
        credit_point = None
        if isinstance(value, int) or isinstance(value, float):
            for interval, woe in self.scorecard_map_dict[feature].items():
                if isinstance(interval, pd.Interval) and value in interval:
                    credit_point = woe
            credit_point = (
                self.scorecard_map_dict[feature].get("Missing", None)
                if credit_point is None
                else credit_point
            )
        elif isinstance(value, str):
            credit_point = self.scorecard_map_dict[feature][value]
        return credit_point

    # Public methods
    def generate_scorecard(self) -> pd.DataFrame:
        model_summary = pd.DataFrame(
            {
                "Characteristic": self.pipeline[
                    LogisticRegression.__name__
                ].feature_names_in_,
                "Estimate": self.pipeline[LogisticRegression.__name__].coef_.reshape(
                    -1
                ),
            }
        )

        self.scorecard = pd.merge(
            left=model_summary,
            right=self.pipeline[WOETransformer.__name__].woe_df,
            how="left",
            on=["Characteristic"],
        )
        self.scorecard["Points"] = (self.offset / self.n_features) - self.factor * (
            (self.scorecard["WOE"] * self.scorecard["Estimate"])
            + (self.intercept / self.n_features)
        )
        self.__generate_mapping(self.scorecard)

    def calculate_single_credit_score(self, data: dict) -> dict:
        details = {
            feature: self.__get_credit_point(feature, value)
            for feature, value in data.items()
        }
        details["credit_score"] = sum(details.values())
        return details

    def calculate_credit_score(self, X: pd.DataFrame) -> pd.DataFrame:
        index = X.index
        results = X.apply(
            lambda instance: self.calculate_single_credit_score(instance.to_dict()),
            axis=1,
        )
        return pd.DataFrame.from_dict(results.tolist()).set_index(index)
