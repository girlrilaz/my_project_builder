import logging.config
import yaml
import time,os,re,csv,sys,uuid,joblib
from datetime import date

today = date.today()
day_folder = f"{today.year}-{today.month}-{today.day}"

if not os.path.exists(os.path.join(".","logs",day_folder)):
    os.mkdir(os.path.join("logs", day_folder))

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

def update_train_log(data_shape, runtime, model_version, model_version_note, subset=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    day_folder = f"{today.year}-{today.month}-{today.day}"

    if subset:
        logfile = os.path.join("logs", day_folder, "model-train-subset.log")
    else:
        logfile = os.path.join("logs", day_folder,f"model-train-{today.year}-{today.month}-{today.day}.log")
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','x_shape','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),data_shape,
                            model_version,model_version_note,runtime])
        writer.writerow(to_write)

def update_evaluation_log(accuracy, roc_auc,runtime,model_version):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    day_folder = f"{today.year}-{today.month}-{today.day}"
    # if test:
    #     logfile = os.path.join("logs","predict-test.log")
    # else:
    logfile = os.path.join("logs", day_folder ,f"model-eval-{today.year}-{today.month}-{today.day}.log")
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','accuracy', 'roc_auc','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),accuracy, roc_auc,
                            model_version,runtime])
        writer.writerow(to_write)


# if __name__ == "__main__":

#     """
#     basic test procedure for logger.py
#     """

#     from src.model import MODEL_VERSION, MODEL_VERSION_NOTE
    
#     ## train logger
#     update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
#                      MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
#     ## predict logger
#     update_predict_log("[0]","[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
#                        "00:00:01",MODEL_VERSION, test=True)