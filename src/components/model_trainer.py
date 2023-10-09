import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

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

class ModelTrainer:
    def __init__(self, agency_name, config):
        self.model_trainer_config = config
        self.agency_name = agency_name
        logging.info(f"\n<<<<<<<<<<<<<<<<<<<<<<<<<< {agency_name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Working on Models!")

            X_train, y_test = train_array, test_array

            models = {
                'ARIMA': ARIMA.ARIMAModel(),
                'SARIMA': SARIMA.SARIMAModel(),
                'AutoReg': AUTOREG.AutoRegModel(),
                'LinearRegression': LinearReg.LinearRegressionModel()
            }

            model_report, model_instance, forecast_val = evaluate_models(X_train, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = model_instance[best_model_name]

            logging.info(f"The best model is {best_model_name} with a score of {best_model_score}")

            logging.info(f"Best found model on both training and testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            result_file_path = os.path.join(self.model_trainer_config.trained_model_text_path, 'agency_score.txt')

            with open(result_file_path, 'w') as result_file:
                result_file.write(f"Best Model Name: {best_model_name}\n")
                result_file.write(f"R2 Score: {best_model_score}\n")

            return best_model, best_model_name, best_model_score

        except Exception as e:
            raise CustomException("No best model found")

    def predict_and_store(self, resampled, model, name):
        if name == 'LinearRegression':
            # Example code for making future predictions with Linear Regression
            target_variable = np.cumsum(np.random.randn(100))
            future_predictions = []
            for i in range(12):
                future_val = np.array([[target_variable[-1], target_variable[-2], target_variable[-3]]])
                future_prediction = model[0].predict(future_val)
                future_predictions.append(future_prediction)
                target_variable = np.append(target_variable, future_prediction)
            print("Future Predictions:", future_predictions)
