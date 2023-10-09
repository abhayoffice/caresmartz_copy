# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:07:27 2023

@author: abhay.bhandari
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

class model_input(BaseModel):
    number_of_predictions : int   


# loading the saved model
agency_model = pickle.load(open('myModel.pkl', 'rb'))    
      
@app.post('/agency_prediction')
def agency_predd(input_parameters : model_input):
   input_data = input_parameters.json()
   input_dictionary = json.loads(input_data)
   num = input_dictionary['number_of_predictions']
   
   prediction = agency_model.forecast(steps =  num)
   
   forecasted_val = prediction.tolist()
   
   print(prediction)
   
   return prediction