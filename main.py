from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.model_trainer_pipeline import ModelTrainerPipeline


STAGE_NAME = 'data Ingestion stage'

try :
    logger.info(f'>>>>>>>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<<<')
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion_pipeline()
    logger.info(f'stage {STAGE_NAME} completed')

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = 'data Transformation stage'

try :
    logger.info(f'>>>>>>>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<<<')
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation_pipeline()
    logger.info(f'stage {STAGE_NAME} completed')

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = 'Model Training stage'

try :
    logger.info(f'>>>>>>>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<<<')
    model_training_pipeline = ModelTrainerPipeline()
    model_training_pipeline.initiate_model_trainer_pipeline()
    logger.info(f'stage {STAGE_NAME} completed')

except Exception as e:
    logger.exception(e)
    raise e