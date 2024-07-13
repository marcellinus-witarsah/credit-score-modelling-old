import pandas as pd
import numpy as np
from src.constants import CREDIT_LEVELS_DESCRIPTIONS
from src.constants import LEVEL_BOUNDS


def interpret_credit_score(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:

    conditions = [
        (df[target_col] > LEVEL_BOUNDS[i]) & (df[target_col] <= LEVEL_BOUNDS[i + 1])
        for i in range(len(LEVEL_BOUNDS) - 1)
    ]

    level_choices = list(CREDIT_LEVELS_DESCRIPTIONS.keys())
    lower_bound_choices = LEVEL_BOUNDS[:-1]
    upper_bound_choices = LEVEL_BOUNDS[1:]
    df["credit_level"] = np.select(conditions, level_choices)
    df["credit_lower_bound"] = np.select(conditions, lower_bound_choices)
    df["credit_upper_bound"] = np.select(conditions, upper_bound_choices)
    df["credit_description"] = df["credit_level"].map(CREDIT_LEVELS_DESCRIPTIONS)
    return df


def generate_report(df: pd.DataFrame) -> pd.DataFrame:
    # Count every Good Users and Bad Users in each credit levels
    report_df = interpret_credit_score(df, target_col="credit_score")
    report_df = (
        report_df.groupby(
            [
                "credit_level",
                "credit_lower_bound",
                "credit_upper_bound",
                "credit_description",
            ]
        )["loan_status"]
        .value_counts()
        .unstack()
        .reset_index()
        .rename(
            columns={
                0: "Good Customers",
                1: "Bad Customers",
                "credit_level": "Credit Level",
                "credit_lower_bound": "Credit Lower Bound",
                "credit_upper_bound": "Credit Upper Bound",
                "credit_description": "Credit Description",
            }
        )
        .fillna(0)
    )

    report_df = report_df.rename_axis(None, axis=1)
    report_df["Customers"] = report_df["Good Customers"] + report_df["Bad Customers"]

    # Proportion of customers exists in each credit levels
    report_df["Customers Rate"] = (
        report_df["Customers"] / report_df["Customers"].sum() * 100
    )

    # Proportion of `Bad Customers`` in each credit levels
    report_df["Default Rate"] = (
        report_df["Bad Customers"] / report_df["Customers"] * 100
    )
    report_df["Reverse Cumulative Customers"] = report_df["Customers"][::-1].cumsum()
    report_df["Reverse Cumulative Good Customers"] = report_df["Good Customers"][
        ::-1
    ].cumsum()
    report_df["Reverse Cumulative Bad Customers"] = report_df["Bad Customers"][
        ::-1
    ].cumsum()

    # Calculate the cumulative percentage of `Good Customers` covered from all customers (start at level 8 down to level 1)
    report_df["Good Customers Coverage"] = (
        report_df["Reverse Cumulative Good Customers"]
        / report_df["Good Customers"].sum()
    )

    # Calculate the cumulative percentage of `Bad Customers` covered from all customers (start at level 8 down to level 1)
    report_df["Loss Coverage"] = (
        report_df["Reverse Cumulative Bad Customers"]
        / report_df["Reverse Cumulative Customers"]
    )

    return report_df
