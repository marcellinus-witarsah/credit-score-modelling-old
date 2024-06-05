from src.utils.common import logger
from src.config.configuration_manager import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.data.data_ingestion import PandasDataIngestionStrategy


class DataIngestionPipeline:
    """
    Class to manage the data ingestion training pipeline.
    """

    def __init__(self):
        """
        Instantiate `DataIngestionPipeline` class.
        """
        self.configuration_manager = ConfigurationManager()

    def run(self):
        """
        Ingest data using the Pandas data ingestion strategy.
        """
        configuration_manager = ConfigurationManager()
        data_ingestion = DataIngestion(
            config=configuration_manager.get_data_ingestion_config()
        )
        data_ingestion.ingest_data(strategy=PandasDataIngestionStrategy())


if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        data_ingestion_training_pipeline = DataIngestionPipeline()
        data_ingestion_training_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.error(e)
