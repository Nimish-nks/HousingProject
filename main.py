from src.housingproject.logger import logging
from src.housingproject.exception import CustomException
from src.housingproject.components.data_ingestion import DataIngestion
import sys

if __name__ =="__main__":
    logging.info("Logging started with main")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

