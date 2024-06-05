from src.utils.common import logger
from src.config.configuration_manager import ConfigurationManager
from src.models.model_training import ModelTraining


class ModelTrainingPipeline:
    """
    Class to manage the model training pipeline.
    """

    def __init__(self):
        """
        Instantiate `ModelTrainingPipeline` class.
        """
        self.configuration_manager = ConfigurationManager()

    def run(self):
        """
        Execute the model training process.
        """
        model_training = ModelTraining(
            config=self.configuration_manager.get_model_training_config()
        )
        model_training.train()


if __name__ == "__main__":
    STAGE_NAME = "Model Training Stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.error(e)
