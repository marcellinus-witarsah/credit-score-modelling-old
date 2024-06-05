import pandas as pd
from src.utils.common import logger
from src.entities.config_entity import DataValidationConfig


class DataValidation:
    """
    Class to handle the data validation process.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Instantiate `DataValidation` class.

        Args:
            config (DataValidationConfig): Configuration for data validation.
        """
        self.config = config

    def validate_data(self):
        """
        Validate the data based on the provided schema.

        This method reads a CSV file from the source path and checks if all columns match the schema.

        Logs messages indicating whether data types match or not.
        """
        try:
            logger.info("Validate data")
            validation_status = None

            df = pd.read_csv(self.config.source_path)
            all_cols = df.columns
            all_schema = self.config.schema

            for col in all_cols:
                if col not in all_schema.keys():
                    validation_status = False
                else:
                    if df[col].dtype == all_schema[col]:
                        validation_status = True
                    else:
                        validation_status = False

            if validation_status:
                logger.info("All data types match")
            else:
                logger.info("There's a data types mismatch")

        except Exception as e:
            logger.error(e)
