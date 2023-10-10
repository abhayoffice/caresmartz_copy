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
    data_transformations = config["data"]

    # Discover agencies based on subfolders in the data directory
    # for agency_name in os.listdir(data_folder):
    logging.info(f"\n<<<<<<<<<<<<<<<<<<<<<<<<<< {data_folder} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    # folder_path = os.path.join(data_folder, agency_name)

    # Initialize DataLoader for each agency
    data_loader = DataLoader(os.path.join(data_folder))

    # Load data
    get_folder = data_loader.get_the_folders(data_transformations)
    # if get_folder not in dicts.items():
    #     dicts[get_folder] = []
    # dicts[get_folder].append(get_folder)

    print("\nfinal output : \n","\t", get_folder)

if __name__ == "__main__":
    main()
