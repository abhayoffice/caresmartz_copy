# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:53:27 2023

@author: abhay.bhandari
"""

#import json
import requests


url = 'http://127.0.0.1:8000/agency_prediction'

input_data_for_model = {
    
    'number_of_predictions' : 8
    
    }

#input_json = json.dumps(input_data_for_model)

response = requests.post(url, json=input_data_for_model)

if response.status_code == 200:
    data = response.json()
    print("Predicted Values:", data)
else:
    print("Error:", response.status_code, response.text)