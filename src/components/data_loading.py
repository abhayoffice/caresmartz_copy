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
    folder_path: str = ""  # Update this default value if needed
    result_dict: dict = None
    raw_data_path: str = os.path.join("agency", "sales.csv")  # Default raw data path


class DataLoader:
    def __init__(self, folder_path):
        self.loader_config = DataLoaderConfig()
        self.subfolder_path = folder_path

    def get_the_folders(self):
        agency_names = []
        csv_file = None
        result_dict = {}
        logging.info("Starting the data loader process....")
        # Assuming folder_path contains subfolders with agency names
        # for subfolder_name in os.listdir(self.folder_path):
        #     subfolder_path = os.path.join(self.folder_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            # Save the agency name (subfolder name) to the list
            agency_names.append(subfolder_name)

            # Load data from sales.csv in each subfolder
            sales_data_path = os.path.join(subfolder_path, 'sales.csv')
            if os.path.exists(sales_data_path):
                csv_file = sales_data_path
                obj = DataIngestion()
                df = pd.read_csv(csv_file)
                result_dict = obj.run_the_files(df, subfolder_name)
                logging.info(f"data loaded successfully for {subfolder_name}")
            else:
                result_dict[subfolder_name] = {"message": "404 - No 'sales.csv found"}
                logging.info(f"No sales.csv for {subfolder_name}")

    self.loader_config.raw_data_path = csv_file
    return agency_names, result_dict

# if __name__ == "__main__":
#     folder_path = r"C:\Users\abhay.bhandari\Jupyter files\Caresmartz360\Agencies"  # Path to the folder where all agencys are stored.
#     data_loader = DataLoader(folder_path)
#
#     # Get the agency names and update data paths
#     agency_names, result_dict = data_loader.get_the_folders()
#     print("===-----------==========================-------------===")
#     print("The agency is : ",agency_names)
#     print("The json results are :",result_dict)
