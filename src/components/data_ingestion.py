import os
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import json

@dataclass
class DataIngestion:
    def __init__(self, config):
        # Initialize DataIngestion with a configuration dictionary.
        self.ingestion_config = config

    def preprocess_data(self, df):
        """
        Preprocess the time series data and change the type of Invoice amount. Also resample the
        data from random values to a 10 days progressive value.
        """
        logging.info("Preprocessing data...")
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        df['Invoice Amount'] = df['Invoice Amount'].str.replace('$', '').str.replace(',', '').astype(float)
        df['Invoice Amount'] = pd.to_numeric(df['Invoice Amount'])
        df.set_index('Invoice Date', inplace=True)

        # Resample the data to 10 days.
        data_10d = df.resample('10D').sum()
        df_10d = pd.DataFrame(data_10d)
        return df_10d

    def initiate_data_ingestion(self, df):
        logging.info("Entering the data ingestion method")

        try:
            logging.info('Reading the dataset as a dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config['train_data_path']), exist_ok=True)

            logging.info("Train-test split initiated")
            df_10d = self.preprocess_data(df)

            # Here we are performing train/test split on the resampled data.
            train_size = int(len(df_10d) * 0.8)
            train_set, test_set = df_10d[:train_size], df_10d[train_size:]

            # Define the file paths using os.path.join
            train_data_file = os.path.join(self.ingestion_config['train_data_path'])
            test_data_file = os.path.join(self.ingestion_config['test_data_path'])
            raw_data_file = os.path.join(self.ingestion_config['raw_data_path'])

            # Store the train_set, test_set, and df_10d as CSV files at the specified locations
            train_set.to_csv(train_data_file, header=True)
            test_set.to_csv(test_data_file, header=True)
            df_10d.to_csv(raw_data_file, header=True)

            logging.info("Data ingestion completed successfully.")
            return train_set, test_set, df_10d

        except Exception as e:
            logging.error("Data ingestion failed.")
            raise CustomException(e)

    def run_the_files(self, df, agency_name, loader_config):
        logging.info("This is the main function that runs the complete data processing pipeline.")
        my_config = loader_config
        try:
            result_dict = {}
            train_data, test_data, raw = self.initiate_data_ingestion(df)
            logging.info("Data ingestion completed, running the data transformation...")
            data_transformation = DataTransformation(my_config)
            train_arr, test_arr, df_arr = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed, initiating model training...")
            model_trainer = ModelTrainer(agency_name, my_config)
            best_model, model_name, model_r2 = model_trainer.initiate_model_trainer(train_arr, test_arr, df_arr)
            result_dict[agency_name] = [model_name, model_r2]
            logging.info("Model training completed successfully.")
            return result_dict

        except Exception as e:
            logging.error("Data processing pipeline failed.")
            raise CustomException(e)
