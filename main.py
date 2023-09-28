from src.housingproject.logger import logging
from src.housingproject.exception import CustomException
import sys

if __name__ =="__main__":
    logging.info("Logging started with main")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

