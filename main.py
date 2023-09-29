from src.housingproject.logger import logging
from src.housingproject.exception import CustomException
from src.housingproject.components.data_ingestion import DataIngestion
from src.housingproject.components.data_transformation import DataTransformation
from src.housingproject.components.model_trainer import ModelTrainer
import sys

if __name__ =="__main__":
    logging.info("Logging started with main")

    try:
        data_ingestion=DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr,test_arr,_ =data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        model_trainer = ModelTrainer()
        #to print R2 score
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

