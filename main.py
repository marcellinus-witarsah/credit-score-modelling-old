from src.utils.common import logger
from src.pipelines.data_ingestion import DataIngestionPipeline
from src.pipelines.data_validation import DataValidationPipeline
from src.pipelines.data_preprocessing import DataPreprocessingPipeline
from src.pipelines.model_training import ModelTrainingPipeline

if __name__ == "__main__":
    STAGE_NAME = "Data ingestion stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(e)

    STAGE_NAME = "Data validation stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(e)

    STAGE_NAME = "Data preprocessing stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        data_preprocessing_pipeline = DataPreprocessingPipeline()
        data_preprocessing_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(e)

    STAGE_NAME = "Model training stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(e)
