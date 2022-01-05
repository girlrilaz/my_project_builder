#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib,yaml
from datetime import date
from dotenv import find_dotenv, load_dotenv

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

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    """
    update train log file
    """

    if not os.path.exists(os.path.join(".","logs", "model_train")):
        os.mkdir(os.path.join(".","logs", "model_train"))

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "model_train", "train-test.log")
    else:
        logfile = os.path.join("logs", "model_train", "train-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','x_shape','eval_test','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), data_shape, eval_test,
                            MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):
    """
    update predict log file
    """
    if not os.path.exists(os.path.join(".","logs", "model_predict")):
        os.mkdir(os.path.join(".","logs", "model_predict"))

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "model_predict", "predict-test.log")
    else:
        logfile = os.path.join("logs", "model_predict", "predict-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), y_pred, y_proba,query,
                            MODEL_VERSION, runtime])
        writer.writerow(to_write)

def update_processing_log(FILENAME, test=False):
    """
    update predict log file
    """
    if not os.path.exists(os.path.join(".","logs", "data_processing")):
        os.mkdir(os.path.join(".","logs", "data_processing"))

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "data_processing", "process-test.log")
    else:
        logfile = os.path.join("logs", "data_processing", "processing-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','filepath']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), FILENAME])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    ## load config data
    # folder to load config file
    CONFIG_PATH = "conf/base"

    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(CONFIG_PATH, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config1 = load_config("catalog.yml")
    config2 = load_config("parameters.yml")

    FILEPATH = os.path.join(config1["base"]["data_path"]["input"], config1["base"]["data"]["filename"])
    MODEL_VERSION = config2["model"]["version"]
    MODEL_VERSION_NOTE = config2["model"]["note"]

    ## train logger
    update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=True)

    ## predict logger
    update_predict_log("[0]", "[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01", MODEL_VERSION, test=True)

    ## processing logger
    update_processing_log(FILEPATH, test=True)

    
    
        
