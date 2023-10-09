import os
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import json


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('agency', 'train.csv')
    test_data_path: str = os.path.join('agency', 'test.csv')
    raw_data_path: str = os.path.join('agency', 'sales.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def preprocess_data(self, df):
        """
        Preprocess the time series data and change the type of Invoice amount.
        """
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        df['Invoice Amount'] = df['Invoice Amount'].str.replace('$', '').str.replace(',', '').astype(float)
        df['Invoice Amount'] = pd.to_numeric(df['Invoice Amount'])
        df.set_index('Invoice Date', inplace=True)
        data_10d = df.resample('10D').sum()
        df_10d = pd.DataFrame(data_10d)
        return df_10d

    def initiate_data_ingestion(self, df):
        logging.info("Entering the data ingestion method")

        try:
            logging.info('Reading the dataset as a dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train-test split initiated")
            df_10d = self.preprocess_data(df)
            train_size = int(len(df_10d) * 0.8)
            train_set, test_set = df_10d[:train_size], df_10d[train_size:]

            train_set.to_csv(self.ingestion_config.train_data_path, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, header=True)
            df_10d.to_csv(self.ingestion_config.raw_data_path, header=True)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e)

    def run_the_files(self, df, agency_name):
        logging.info("This is the main function that runs the complete mid process.")
        try:
            result_dict = {}
            train_data, test_data = self.initiate_data_ingestion(df)
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)
            model_trainer = ModelTrainer(agency_name)
            best_model, model_name, model_r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

            result_dict[agency_name] = [model_name, model_r2]

            return result_dict

        except Exception as e:
            raise CustomException(e)
