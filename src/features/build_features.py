import os, yaml
import logging
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'

def load_data():

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
    logger = logging.getLogger(__name__)
    logger.info(f"{bcolors.WARNING}Data Processing{bcolors.ENDC} : building data features for '{filename}' completed")

    data_dir = os.path.join(config["base"]["data_path"]["output"], config["base"]["data"]["filename"])
    df = pd.read_csv(data_dir)
       
    ## pull out the target and remove uneeded columns
    _y = df.pop('species')
    df.head()
    X = df

    return(X, _y)

if __name__ == "__main__":
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    print(load_data())