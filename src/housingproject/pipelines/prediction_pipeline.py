import os
import sys
import sklearn
import pandas as pd
from src.housingproject.exception import CustomException
from src.housingproject.logger import logging
from src.housingproject.utils import load_object
import pickle

def makePrediction(longitude,latitude,housing_median_age,total_rooms,
                     total_bedrooms,population,households,median_income,
                     ocean_proximity):
    try:
        #convert data to dataframe to get it predicted by ML model
        input_data = {'longitude':[longitude],'latitude':[latitude],'housing_median_age':[housing_median_age],
                      'total_rooms':[total_rooms],'total_bedrooms':[total_bedrooms],'population':[population],'households':[households],
                      'median_income':[median_income],'ocean_proximity':[ocean_proximity]}
        
        input_dict=pd.DataFrame(input_data)
        
        logging.info('Data converted to dataframe for prediction')

        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        logging.info("Before Loading object files of model")
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        logging.info("After Loading object files of model")
        data_scaled=preprocessor.transform(input_dict)
        preds=model.predict(data_scaled)
        return preds
        
    except Exception as e:
        raise CustomException(e,sys)

