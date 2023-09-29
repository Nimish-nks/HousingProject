#using gradio framework

from src.housingproject.exception import CustomException
from src.housingproject.logger import logging
from src.housingproject.pipelines.prediction_pipeline import makePrediction
import pandas as pd
import gradio as gr



try:
    logging.info("Taking data from user")
    with gr.Blocks() as housing_app:

        longitude=gr.Number(label="Enter longitude")
        latitude=gr.Number(label="Enter latitude")
        housing=gr.Number(label="Enter housing")
        housing_median_age=gr.Number(label="Enter housing median age")
        total_rooms=gr.Number(label="Enter total number of room")
        total_bedrooms=gr.Number(label="Enter total number of bedroom")
        median_income=gr.Number(label="Enter median income")
        population=gr.Number(label="Enter population")
        households=gr.Number(label="Enter households")
        
        ocean_proximity=gr.Radio(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], 
                                 label="Ocean_Proximity", info="What is your's?")
        
        logging.info("Data from user taken")
        button = gr.Button(value='Predict median house value')

        #convert data to dataframe to get it predicted by ML model
        input_data = {'longitude':[longitude],'latitude':[latitude],'housing_median_age':[housing_median_age],
                      'total_rooms':[total_rooms],'total_bedrooms':[total_bedrooms],'population':[population],'households':[households],
                      'median_income':[median_income],'ocean_proximity':[ocean_proximity]}
        
        input_dict=pd.DataFrame(input_data)
        
        logging.info('Data converted to dataframe for prediction')

        housing_median_value = makePrediction(input_dict)  

        button.click(fn=makePrediction, 
                    inputs=[longitude,latitude,housing_median_age,total_rooms,
                     total_bedrooms,population,households,median_income,
                     ocean_proximity],
                    outputs=housing_median_value)

    housing_app.launch()
except Exception as e:
    raise CustomException("Cannot launch housing gradio app")



