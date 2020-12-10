import time,os,re,csv,sys,uuid,joblib, yaml
import pickle
from datetime import date
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from src.logger import update_predict_log, update_train_log
from src.features.build_features import load_data

## load environment variables
load_dotenv(find_dotenv())

## load config data
# folder to load config file
CONFIG_PATH = "conf/base"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name), 'r') as file:
        config = yaml.safe_load(file)

    return config

model_config = load_config("parameters.yml")

## load model parameters from conf/base/parameters.yml
MODEL_VERSION = model_config["model"]["version"]
MODEL_VERSION_NOTE = model_config["model"]["note"]
SAVED_MODEL = os.path.join("models", "model-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION))))

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models") 

def model_predict(query, model=None, test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()
    
    ## input checks
    if isinstance(query, dict):
        query = pd.DataFrame(query)
    elif isinstance(query, pd.DataFrame):
        pass
    else:
        raise Exception("ERROR (model_predict) - invalid input. {} was given".format(type(query)))

    ## features check
    features = sorted(query.columns.tolist())
    if features != ['petal_length', 'petal_width', 'sepal_length', 'sepal_width', ]:
        print("query features: {}".format(",".join(features)))
        raise Exception("ERROR (model_predict) - invalid features present") 
    
    ## load model if needed
    if not model:
        model = model_load()
    
    ## output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = 'None'
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    for i in range(query.shape[0]):
        update_predict_log(y_pred[i], y_proba, query.iloc[i].values.tolist(), 
                           runtime, MODEL_VERSION, test=test)
        
    return({'y_pred':y_pred, 'y_proba':y_proba})

def model_load(test=False):
    """
    example funtion to load model
    """
    if test : 
        print( "... loading test version of model" )
        model = joblib.load(os.path.join("models","test.joblib"))
        return(model)

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)

if __name__ == '__main__':

    """
    basic test procedure for predict_model.py
    """

    ## load the model
    model = model_load(test=True)
    
    ## example predict
    query = pd.DataFrame({'sepal_length': [5.1, 6.4, 6.9],
                          'sepal_width': [3.5, 3.2, 3.2],
                          'petal_length': [1.4, 4.5, 5.7],
                          'petal_width': [0.2, 1.5, 2.3]
    })

    result = model_predict(query, model, test=True)
    y_pred = result['y_pred']
    print("predicted: {}".format(y_pred))