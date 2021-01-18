# -*- coding: utf-8 -*-
import os
import glob
import yaml
import logging
import pandas as pd
from pathlib import Path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main(filename, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Input and output path are saved in catalog.yml file in conf/base folder. Please create one and
        save it under the project root directory if it does not exist
    """
        
    logger = logging.getLogger(__name__)
    logger.info(f"{bcolors.WARNING}Data Processing{bcolors.ENDC} : data cleaning and transformation for '{filename}' from folder data/raw to data/processed completed")

    # Read data from input path
    input_path = input_filepath
    df = pd.read_csv(input_path)

    # TODO: process data here

    # Save data to output path
    output_path = output_filepath
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    ## load config data
    # folder to load config file
    CONFIG_PATH = "conf/base"

    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(CONFIG_PATH, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config = load_config("catalog.yml")
    
    filename = config["base"]["data"]["filename"]
    filepath = os.path.join(config["base"]["data_path"]["input"], config["base"]["data"]["filename"])
    input_path = os.path.join(config["base"]["data_path"]["input"], config["base"]["data"]["filename"])
    output_path = os.path.join(config["base"]["data_path"]["output"], config["base"]["data"]["filename"])

    main(filename, input_path, output_path)
