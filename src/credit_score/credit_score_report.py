import pandas as pd
import numpy as np
from src.constants import LEVEL_BOUNDS
from src.constants import CREDIT_LEVELS_DESCRIPTIONS


def interpret_credit_score(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Interpret the credit score into predefined levels and descriptions.

    Args:
        df (pd.DataFrame): The dataframe containing credit score data.
        target_col (str): The column name for the credit scores in the dataframe.

    Returns:
        pd.DataFrame: The dataframe with additional columns for credit level, bounds, and descriptions.
    """
    # 1. Define conditions to categorize credit scores based on LEVEL_BOUNDS
    conditions = [
        (df[target_col] > LEVEL_BOUNDS[i]) & (df[target_col] <= LEVEL_BOUNDS[i + 1])
        for i in range(len(LEVEL_BOUNDS) - 1)
    ]

    # 2. Prepare choices for credit levels, lower bounds, and upper bounds
    level_choices = list(CREDIT_LEVELS_DESCRIPTIONS.keys())
    lower_bound_choices = LEVEL_BOUNDS[:-1]
    upper_bound_choices = LEVEL_BOUNDS[1:]

    # 3. Assign credit level, lower bound, upper bound, and description based on conditions
    df["credit_level"] = np.select(conditions, level_choices)
    df["credit_lower_bound"] = np.select(conditions, lower_bound_choices)
    df["credit_upper_bound"] = np.select(conditions, upper_bound_choices)
    df["credit_description"] = df["credit_level"].map(CREDIT_LEVELS_DESCRIPTIONS)
    return df


def generate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a report categorizing customers into credit levels and calculating the summary statistics (cumulative percentage, proportion, etc).

    Args:
        df (pd.DataFrame): The dataframe containing customer credit scores.

    Returns:
        pd.DataFrame: The report dataframe with credit levels, customer counts, and various statistics.
    """

    # 1. Categorize into credit level based on credit score:
    report_df = interpret_credit_score(df, target_col="credit_score")

    # 2. Count `Good customers` and `Bad customers` in each credit level:
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

    # 3. Calculate proportion of customers in each credit level
    report_df["Customers Rate"] = (
        report_df["Customers"] / report_df["Customers"].sum() * 100
    )

    # 4. Proportion of `Bad Customers` in each credit level
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

    # 5. Calculate the cumulative percentage of `Good Customers` from all customers
    report_df["Good Customers Coverage"] = (
        report_df["Reverse Cumulative Good Customers"]
        / report_df["Good Customers"].sum()
    )

    # 6. Calculate the cumulative percentage of `Bad Customers` from all customers
    report_df["Loss Coverage"] = (
        report_df["Reverse Cumulative Bad Customers"]
        / report_df["Reverse Cumulative Customers"]
    )

    return report_df
