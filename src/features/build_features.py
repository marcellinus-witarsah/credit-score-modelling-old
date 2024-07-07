# def generate_bins(self) -> pd.DataFrame:
#     """
#     Create bins for a numerical column, dividing it into a specified number of equal-sized bins.

#     Args:
#         df (pd.DataFrame): Pandas DataFrame containing the data.
#         numerical_column (str): Numerical column.
#         num_of_bins (int): Number of bins to create.
#     Returns:
#         pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
#     """
#     # 1. Load training data:
#     df = pd.read_csv(self.config.train_file)
#     for numerical_column in df.select_dtypes("number"):
#         if numerical_column != self.config.target:
#             df[numerical_column] = pd.qcut(
#                 df[numerical_column], q=self.config.num_of_bins, duplicates="drop"
#             )
#     return df

# def fill_missing_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Fill missing categorical columns inside Pandas DataFrame with `Missing`.

#     Args:
#         df (pd.DataFrame): Pandas DataFrame containing the data.
#     Returns:
#         pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
#     """
#     for column in df.columns:
#         if df[column].isna().sum() > 0 and df[column].dtype in [
#             "object",
#             "category",
#         ]:
#             # Add category 'Missing' to replace the missing values
#             df[column] = df[column].cat.add_categories("Missing")

#             # Replace missing values with category 'Missing'
#             df[column] = df[column].fillna(value="Missing")
#     return df
