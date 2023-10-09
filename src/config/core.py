import os
import yaml
from src.components.data_loading import DataLoader
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object
from src.logger import logging


def main():
    # Get the absolute path to the 'src' directory
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct the path to the 'config.yml' file in the 'src' directory
    config_file_path = os.path.join(src_dir, "config.yml")
    # Load the configuration from config.yml
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get the data folder from the config
    data_folder = config["data"]["data_folder"]

    # Discover agencies based on subfolders in the data directory
    agency_names = []


    for agency_name in os.listdir(data_folder):
        logging.info(f"\n<<<<<<<<<<<<<<<<<<<<<<<<<< {agency_name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        print(f"======={agency_name}======")
        subfolder_path = os.path.join(data_folder, agency_name)

        # Initialize DataLoader, DataTransformation, and ModelTrainer for each agency
        data_loader = DataLoader(os.path.join(data_folder, agency_name))
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer(agency_name)

        # Load data
        train_data, test_data = data_loader.get_train_test_data()

        # Perform data transformation
        train_data_transformed, test_data_transformed = data_transformation.initiate_data_transformation(train_data,
                                                                                                         test_data)

        # Train and evaluate models
        best_model, best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_data_transformed,
                                                                                             test_data_transformed)

        # Save the best model
        model_file_path = os.path.join(config["model"]["save_folder"], agency_name,
                                       config["model"]["trained_model_file"])
        load_object(file_path=model_file_path, obj=best_model)

if __name__ == "__main__":
    main()
