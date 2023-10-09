import os
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)

def evaluate_models(train, test, models):
    try:
        report = {}
        my_instance = {}

        for model_name, model in models.items():
            model_instance = model.get_model(train, test)  # Instantiate the model

            # Get the forecasting value.
            forecast_instance = model.get_forecast(train, test, model_instance)

            # Train the model on the training data and calculate the R^2 score.
            r2_score_value = model.get_score(test, forecast_instance)
            print(f"The R^2 score for {model_name} is {r2_score_value}")

            # Store the R^2 score in the report
            if model_name not in report:
                report[model_name] = []
                my_instance[model_name] = []

            report[model_name].append(r2_score_value)
            my_instance[model_name].append(model_instance)

        return report, my_instance, forecast_instance

    except Exception as e:
        raise CustomException(e)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e)
