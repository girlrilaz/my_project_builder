import time,os,re,csv,sys,uuid,joblib
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

## load data from build_features
data = load_data()

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models") 

model_version = os.getenv("MODEL_VERSION")
model_version_note = os.getenv("MODEL_VERSION_NOTE")
SAVED_MODEL = os.path.join("models", "model-{}.joblib".format(re.sub("\.", "_", str(model_version))))

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

print(data)
print(type(data))