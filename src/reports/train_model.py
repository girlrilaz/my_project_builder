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

def get_preprocessor():
    """
    return the preprocessing pipeline
    """

    ## preprocessing pipeline
    numeric_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])

    categorical_features = []
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    return(preprocessor)

def model_train(test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    The iris dataset is already small so the subset is shown as an example

    Note that the latest training data is always saved to be used by perfromance monitoring tools.
    """
    
    ## load model parameters from conf/base/parameters.yml
    model_config = load_config("parameters.yml")
    test_size = model_config["model"]["test_size"]
    n_est = model_config["classifier"]["param_grid"]["clf__n_estimators"]
    criterion = model_config["classifier"]["param_grid"]["clf__criterion"]
    max_depth = model_config["classifier"]["param_grid"]["clf__max_depth"]

    ## start timer for runtime
    time_start = time.time()
    
    ## data ingestion from build_features
    X, y = load_data()

    preprocessor = get_preprocessor()

    ## subset the data to enable faster unittests
    if test:
        n_samples = int(np.round(0.9 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples, replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]  
    
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    ## Specify parameters and model
    param_grid = {
        'clf__n_estimators': n_est,
        'clf__criterion':criterion,
        'clf__max_depth':max_depth
    }

    print("... grid searching")
    clf = ensemble.RandomForestClassifier()
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    params = grid.best_params_
    params = {re.sub("clf__","",key):value for key,value in params.items()}
    
    ## fit model on training data
    clf = ensemble.RandomForestClassifier(**params)
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    eval_test = classification_report(y_test, y_pred, output_dict=True)
    
    ## retrain using all data
    pipe.fit(X, y)

    if test:
        print("... saving test version of model")
        joblib.dump(pipe, os.path.join("models", "test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        joblib.dump(pipe, SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models", 'latest-train.pickle')
        with open(data_file, 'wb') as tmp:
            pickle.dump({'y':y, 'X':X}, tmp)
        
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_train_log(X.shape, eval_test, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

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
    basic test procedure for train_model.py
    """
    
    ## train the model
    model_train(test=False)