from src.housingproject.logger import logging
from src.housingproject.exception import CustomException
from src.housingproject.components.data_ingestion import DataIngestion
from src.housingproject.components.data_transformation import DataTransformation
import sys

if __name__ =="__main__":
    logging.info("Logging started with main")

    try:
        data_ingestion=DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transormation(train_data_path,test_data_path)
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

