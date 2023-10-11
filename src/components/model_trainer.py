# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:58:39 2023

@author: abhay.bhandari
"""
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.Models import ExpoSmoothing
from src.exception import CustomException
from src.logger import logging
import src.Models.ARIMA_model as ARIMA
import src.Models.SARIMA_model as SARIMA
import src.Models.Auto_Reg_model as AUTOREG
import src.Models.LinearRegression as LinearReg
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    agency_name: str
    trained_model_file_path: str = None  # Initialize it as None

    def __post_init__(self):
        self.trained_model_file_path = os.path.join("artifacts", self.agency_name, f"{self.agency_name}.pkl")
        self.trained_model_text_path = os.path.join("artifacts", self.agency_name)
        self.trained_model_csv_file = os.path.join("artifacts", self.agency_name, "prediction")


class ModelTrainer:
    def __init__(self, agency_name, config):
        #This config is static and is used to store the predictions and pickle files.
        self.model_trainer_config = ModelTrainerConfig(agency_name)

        #This config is dynamic and is called from the config.yml file for changing the start_date while calculating predictions.
        self.external_config = config
        self.agency_name = agency_name
        print(f"\n<<<<<<<<<<<<<<<<<<<<<<<<<< {agency_name}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    def initiate_model_trainer(self, train_array, test_array, data):
        try:
            logging.info("Working on Models !")

            # X_train,y_test= train_array[:,:-1],test_array[:,:-1]
            X_train, y_test = train_array, test_array

            models = {
                'ARIMA': ARIMA.ARIMAModel(),
                'SARIMA': SARIMA.SARIMAModel(),
                'AutoReg': AUTOREG.AutoRegModel(),
                'LinearRegression': LinearReg.LinearRegressionModel(),
                'ExpoSmoothing': ExpoSmoothing.ESModel()
            }

            # model_report , model_instance , forecast_val = evaluate_models(X_train, y_test, models )
            model_report, model_instance = evaluate_models(X_train, y_test, models)
            print(f"The dataType of best score is : ", model_report.items(), " and the key is ", model_report.values())

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Choose the best model here
            best_model = model_instance[best_model_name]

            print(f"The best model is {best_model_name} and it's score is {best_model_score}")

            logging.info(f"Best found model on both training and testing dataset")

            # if float(best_model_score[0]) < 0.0:
            #     raise CustomException("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            result_file_path = os.path.join(self.model_trainer_config.trained_model_text_path, 'agency_score.txt')

            with open(result_file_path, 'w') as result_file:
                result_file.write(f"Best Model Name: {best_model_name}\n")
                result_file.write(f"R2 Score: {best_model_score}\n")

            # Call the prediction method for storing the data.
            self.predict_and_store(data, best_model, best_model_name)

            return best_model, best_model_name, best_model_score

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise CustomException("No best model found")

    def predict_and_store(self, resampled, model, name):

        global future_pred
        print("Inside predict_and_store :", model[0], " and the name is ", name)
        start_date = pd.Timestamp('2023-09-01')

        if name == 'LinearRegression':
            target_variable = np.cumsum(np.random.randn(100))
            future_predictions = []

            for i in range(12):
                future_val = np.array([[target_variable[-1], target_variable[-2], target_variable[-3]]])
                future_prediction = model[0].predict(future_val)
                future_predictions.append(future_prediction)
                target_variable = np.append(target_variable, future_prediction)

            # Reshape future_predictions to 2D array
            future_pred = np.array(future_predictions).flatten()
            # future_predictions = np.array(future_predictions).reshape(-1,1)
            print("-----------future predictions------------\n",future_pred)
            # future_pred = pd.DataFrame(future_predictions)

        elif name == 'ARIMA':
            future_pred = model[0].predict(steps=12)

        elif name == 'SARIMA':
            future_pred = model[0].forecast(steps=12)

        elif name == 'AutoReg':
            future_pred = model[0].predict(start=len(resampled), end=len(resampled) + 12, dynamic=False)
        elif name == 'ExpoSmoothing':
            future_pred = model[0].forecast(steps=12)
        else:
            print("No future prediction was made")
            return

        # Create a temporary DataFrame with invoice date and invoice amount columns
        # start_date = pd.Timestamp('2023-09-01')
        start_date = pd.Timestamp(self.external_config['start_date'])
        end_date = start_date + pd.DateOffset(months= len(future_pred))
        date_range = pd.date_range(start_date, end_date, freq='M')
        temp_df = pd.DataFrame({'invoice date': date_range, 'invoice amount': future_pred})


        # Check if the trained_model_csv_file directory exists, and create it if not
        os.makedirs(self.model_trainer_config.trained_model_csv_file, exist_ok=True)

        # Save the future_pred DataFrame as a CSV file
        csv_file_path = os.path.join(self.model_trainer_config.trained_model_csv_file, 'forecasts.csv')
        temp_df.to_csv(csv_file_path, index=False)
        print(f"Future predictions saved as CSV: {csv_file_path}")



