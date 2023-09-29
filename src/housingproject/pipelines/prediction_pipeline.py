import os
import sys
from src.housingproject.exception import CustomException
from src.housingproject.logger import logging
from src.housingproject.utils import load_object
import pickle

def makePrediction(features):
    try:
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        logging.info("Before Loading object files of model")
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        logging.info("After Loading object files of model")
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        return preds
        
    except Exception as e:
        raise CustomException(e,sys)

