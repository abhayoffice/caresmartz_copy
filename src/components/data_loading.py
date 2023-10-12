import os
import sys
import numpy as np
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    raw_data_path: str = ""  # Default raw data path


class DataLoader:
    folder_path: str = ""  # Update this default value if needed

    def __init__(self, folder_path):
        self.loader_config = DataLoaderConfig()
        self.folder_path = folder_path

    def get_the_folders(self, config):
        agency_names = []
        csv_file = None
        result_dict = []
        logging.info("Starting the data loader process....")

        # Assuming folder_path contains subfolders with agency names
        for subfolder_name in os.listdir(self.folder_path):
            logging.info("Now working on each subfolder.")
            subfolder_path = os.path.join(self.folder_path, subfolder_name)

            # If the subfolder is present, then this if statement is executed.
            if os.path.isdir(subfolder_path):
                # Save the agency name (subfolder name) to the list
                agency_names.append(subfolder_name)
                logging.info(f"Started with the data loading process for {subfolder_name}")

                # Load data from sales.csv in each subfolder
                sales_data_path = os.path.join(subfolder_path, 'sales.csv')
                if os.path.exists(sales_data_path):
                    csv_file = sales_data_path
                    obj = DataIngestion(config)
                    df = pd.read_csv(csv_file)
                    result_dict.append(obj.run_the_files(df, subfolder_name, config))
                    logging.info(f"Data loaded successfully for {subfolder_name}")
                else:
                    logging.info(f"No sales.csv found for {subfolder_name}")
                    result_dict.append({"message": "404 - No 'sales.csv' found"})

        self.loader_config.raw_data_path = csv_file
        return result_dict
