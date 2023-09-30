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
        housing_median_age=gr.Number(label="Enter housing median age")
        total_rooms=gr.Number(label="Enter total number of room")
        total_bedrooms=gr.Number(label="Enter total number of bedroom")
        median_income=gr.Number(label="Enter median income")
        population=gr.Number(label="Enter population")
        households=gr.Number(label="Enter households")
        
        ocean_proximity=gr.Radio(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], 
                                 label="Ocean_Proximity", info="What is your's?")
        output = gr.Textbox(label='Predicted Median House Value')
        logging.info("Data from user taken")
        button = gr.Button(value='Predict median house value')      

        button.click(fn=makePrediction, 
                    inputs=[longitude,latitude,housing_median_age,total_rooms,
                     total_bedrooms,population,households,median_income,
                     ocean_proximity],
                    outputs=output)

    housing_app.launch()
except Exception as e:
    raise CustomException("Cannot launch housing gradio app")



