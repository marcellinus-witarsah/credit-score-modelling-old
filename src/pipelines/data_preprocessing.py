from src.utils.common import logger
from src.config.configuration_manager import ConfigurationManager
from src.data.data_preprocessing import DataPreprocessing


class DataPreprocessingPipeline:
    def __init__(self):
        """
        Instantiate `DataPreprocessingPipeline` class
        """
        self.configuration_manager = ConfigurationManager()

    def run(self):
        """
        Preprocess data
        """
        data_preprocessing = DataPreprocessing(
            config=self.configuration_manager.get_data_preprocessing_config()
        )
        data_preprocessing.split_data()


if __name__ == "__main__":
    STAGE_NAME = "Data Preprocessing Stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        data_preprocessing_pipeline = DataPreprocessingPipeline()
        data_preprocessing_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.error(e)
