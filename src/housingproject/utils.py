import os
import sys
from src.housingproject.exception import CustomException
from src.housingproject.logger import logging
import pandas as pd
from dotenv import load_dotenv

import pymysql

import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')



def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('Select * from housing',mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e,sys)
    
#function to make pickle file of object 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        logging.info("evaluate models called")
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            logging.info("model & param selected")

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            logging.info("hyperparameters tuned")

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            logging.info("model fit over")

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            logging.info("model predict over")

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            logging.info("r2 score over")

            report[list(models.keys())[i]] = test_model_score
            logging.info("evaluate models call ended")

        return report

    except Exception as e:
        raise CustomException(e, sys)