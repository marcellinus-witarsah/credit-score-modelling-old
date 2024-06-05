from src.utils.common import logger
from src.config.configuration_manager import ConfigurationManager
from src.data.data_validation import DataValidation


class DataValidationPipeline:
    def __init__(self):
        """
        Instantiate `DataValidationPipeline` class
        """
        self.configuration_manager = ConfigurationManager()

    def run(self):
        """
        Validate data
        """
        data_validation = DataValidation(
            config=self.configuration_manager.get_data_validation_config()
        )
        data_validation.validate_data()


if __name__ == "__main__":
    STAGE_NAME = "Data Validation Stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.error(e)
