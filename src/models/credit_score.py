import pandas as pd
import numpy as np
from typing import Union
from sklearn.linear_model import LogisticRegression
from src.features.woe_transformer import WOETransformer


class CreditScore:
    def __init__(
        self,
        pipeline,
        pdo: Union[int, float],
        odds: Union[int, float],
        scorecard_points: Union[int, float],
    ):
        """Initialize the CreditScore class with pipeline, PDO, odds, and scorecard points.

        Args:
            pipeline (Pipeline): A scikit-learn pipeline containing the WOE transformer and Logistic Regression.
            pdo (Union[int, float]): Points to Double the Odds, a measure of credit score sensitivity.
            odds (Union[int, float]): Base odds at the scorecard points.
            scorecard_points (Union[int, float]): The score corresponding to the base odds.
        """
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
        self.credit_levels_decriptions = {
            1: "Very Poor",
            2: "Poor",
            3: "Below Average",
            4: "Average",
            5: "Above Average",
            6: "Good",
            7: "Very Good",
            8: "Excellent",
            9: "Exceptional",
        }

    def __generate_mapping(self, scorecard: pd.DataFrame) -> None:
        """Generate a mapping dictionary from scorecard data.

        Args:
            scorecard (pd.DataFrame): A DataFrame containing scorecard characteristics, values, and points.
        """
        scorecard_features = scorecard["Characteristic"].unique()

        for feature in scorecard_features:
            self.scorecard_map_dict[feature] = {
                row["Value"]: row["Points"]
                for _, row in scorecard[
                    scorecard["Characteristic"] == feature
                ].iterrows()
            }

    def __get_credit_point(self, feature: str, value: Union[int, float, str]) -> float:
        """Get the credit point for a given feature and value.

        Args:
            feature (str): The feature name.
            value (Union[int, float, str]): The value of the feature.

        Returns:
            float: The corresponding credit point for the feature value.
        """
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

    def generate_scorecard(self) -> pd.DataFrame:
        """Generate a scorecard DataFrame from the pipeline's logistic regression and WOE transformer.

        Returns:
            pd.DataFrame: A DataFrame containing characteristics, estimates, WOE values, and points.
        """
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
        """Calculate the credit score for a single instance.

        Args:
            data (dict): A dictionary containing feature values for a single instance.

        Returns:
            dict: A dictionary containing the credit points for each feature and the total credit score.
        """
        details = {
            feature: self.__get_credit_point(feature, value)
            for feature, value in data.items()
        }
        details["credit_score"] = sum(details.values())
        return details

    def calculate_credit_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate credit scores for multiple instances.

        Args:
            X (pd.DataFrame): A DataFrame containing feature values for multiple instances.

        Returns:
            pd.DataFrame: A DataFrame with the calculated credit points and total credit score for each instance.
        """
        index = X.index
        results = X.apply(
            lambda instance: self.calculate_single_credit_score(instance.to_dict()),
            axis=1,
        )
        return pd.DataFrame.from_dict(results.tolist()).set_index(index)

    def interpret_credit_score(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """
        Interpret the calculated credit scores by assigning credit levels and descriptions.

        Args:
            df (pd.DataFrame): A DataFrame containing the calculated credit scores.
            target_col (str): The column name containing the credit scores.

        Returns:
            pd.DataFrame: A DataFrame with credit levels, bounds, and descriptions added.
        """
        left_bound = -np.inf
        level_1 = 350
        level_2 = 400
        level_3 = 450
        level_4 = 500
        level_5 = 550
        level_6 = 600
        level_7 = 650
        level_8 = 700
        right_bound = np.inf
        conditions = [
            (df[target_col] > left_bound) & (df[target_col] <= level_1),
            (df[target_col] > level_1) & (df[target_col] <= level_2),
            (df[target_col] > level_2) & (df[target_col] <= level_3),
            (df[target_col] > level_3) & (df[target_col] <= level_4),
            (df[target_col] > level_4) & (df[target_col] <= level_5),
            (df[target_col] > level_5) & (df[target_col] <= level_6),
            (df[target_col] > level_6) & (df[target_col] <= level_7),
            (df[target_col] > level_7) & (df[target_col] <= level_8),
            (df[target_col] > level_8) & (df[target_col] <= right_bound),
        ]

        level_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        lower_bound_choices = [
            left_bound,
            level_1,
            level_2,
            level_3,
            level_4,
            level_5,
            level_6,
            level_7,
            level_8,
        ]
        upper_bound_choices = [
            level_1,
            level_2,
            level_3,
            level_4,
            level_5,
            level_6,
            level_7,
            level_8,
            right_bound,
        ]
        df["credit_level"] = np.select(conditions, level_choices)
        df["credit_lower_bound"] = np.select(conditions, lower_bound_choices)
        df["credit_upper_bound"] = np.select(conditions, upper_bound_choices)
        df["credit_description"] = df["credit_level"].map(
            self.credit_levels_decriptions
        )
        return df
