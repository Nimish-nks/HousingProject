import os
import sys
from src.housingproject.exception import CustomException
from src.housingproject.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.housingproject.utils import save_object
import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    #define function which will perform feature engineering
    def get_data_transformer_object(self):
        logging.info("Data transformation started")

        try:

            numerical_columns=['longitude','latitude','housing_median_age',
                               'total_rooms','total_bedrooms','population',
                               'households','median_income']
            categorical_column=['ocean_proximity']

            logging.info('Handling numerical features')
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])

            logging.info('Handling categorical features')

            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )

            logging.info('preprocessor object created')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transormation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            #making object of get_data_transformer_object which will call pipelines
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= 'median_house_value'
            numerical_columns = ["longitude","latitude","housing_median_age","total_rooms",
                                 "total_bedrooms","population","households","median_income"]
            
            
            

            logging.info("Making target and input sets")
            

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")
            

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            #only transform the test data and not fit
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object")

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

        