import logging.config
import yaml
import time,os,re,csv,sys,uuid,joblib
from datetime import date

with open('configs/yaml/logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger

# if not os.path.exists(os.path.join(".","logs")):
#     os.mkdir("logs")

# def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE):
#     """
#     update train log file
#     """

#     ## name the logfile using something that cycles with date (day, month, year)    
#     today = date.today()

#     logfile = os.path.join("logs","model-train-{today.year}-{today.month}.log")
        
#     ## write the data to a csv file    
#     header = ['unique_id','timestamp','x_shape','eval_test','model_version',
#               'model_version_note','runtime']
#     write_header = False
#     if not os.path.exists(logfile):
#         write_header = True
#     with open(logfile,'a') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',')
#         if write_header:
#             writer.writerow(header)

#         to_write = map(str,[uuid.uuid4(),time.time(),data_shape,eval_test,
#                             MODEL_VERSION,MODEL_VERSION_NOTE,runtime])
#         writer.writerow(to_write)