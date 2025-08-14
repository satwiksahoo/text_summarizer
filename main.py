from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline


STAGE_NAME = 'data Ingestion stage'

try :
    logger.info(f'>>>>>>>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<<<')
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion_pipeline()
    logger.info(f'stage {STAGE_NAME} completed')

except Exception as e:
    logger.exception(e)
    raise e
    