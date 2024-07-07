import time
import pandas as pd
from src.utils.common import logger
from src.config.configuration_manager import ConfigurationManager


class DataValidation:
    """
    Class to handle the data validation process.
    """

    def __init__(self):
        """
        Instantiate `DataValidation` class.
        """
        self.config = ConfigurationManager().data_validation_config

    def run(self):
        """
        Validate the data based on the provided schema.

        This method reads a CSV file from the source path and checks if all columns match the schema.

        Logs messages indicating whether data types match or not.
        """
        try:
            start_time = time.perf_counter()
            logger.info("Validate data")

            # 1. Load data:
            df = pd.read_csv(self.config.raw_data_file)

            # 2. Set up variables for validation
            columns = df.columns
            schema = self.config.schema
            validation_status = None

            # 3. Perform validation by comparing columns from the data and the schema
            for col in columns:
                if col not in schema.keys():
                    validation_status = False
                else:
                    if df[col].dtype == schema[col]:
                        validation_status = True
                    else:
                        validation_status = False

            # 4. Check validation status
            if validation_status:
                logger.info("All data types match")
            else:
                logger.info("There's a data types mismatch")

            elapsed_time = time.perf_counter() - start_time
            logger.info(
                "Validate data finished in {:.2f} seconds.".format(elapsed_time)
            )
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    data_validation = DataValidation()
    data_validation.run()
