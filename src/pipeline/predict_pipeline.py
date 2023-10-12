import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import yaml
import os

class PredictPipeline:
    def __init__(self):
        # Get the absolute path to the 'src' directory
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_file_path = os.path.join(src_dir, "config.yml")
        # Load the configuration from config.yml
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        self.config = config\

    def predict(self, dataframe):
        preprocessor_loc = self.config["data"]["preprocessor_file"]
        preprocessor = load_object(preprocessor_loc)
        df_scaled = preprocessor.fit_transform(dataframe)

    def models_in_dict(self):
        '''
                     For models we have multiple pickle files and to execute each of these files
                     All the models are named according to their algorithm.
        '''
        try:
            # Get the data folder from the config
            models_loc = self.config["data"]["trained_models_root"]

            # Let us store each agency's pkl in a dictionary.
            my_model_collection = {}

            # Assuming folder_path contains subfolders with agency names
            for subfolder_name in os.listdir(models_loc):
                subfolder_path = os.path.join(models_loc, subfolder_name)

                model_pkl_path = os.path.join(subfolder_path, f"{subfolder_name}.pkl")

                my_model = load_object(model_pkl_path)
                if subfolder_name not in my_model_collection.items():
                    my_model_collection[subfolder_name] = my_model

            #Dictionary with all models is returned.
            return my_model_collection

        except Exception as e:
            raise CustomException(e)

class CustomData:
    def __init__(self):
        pass