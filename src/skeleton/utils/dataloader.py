# -*- coding: utf-8 -*-
"""Data Loader"""

# standard library
import sys
import pandas as pd

# external
#import jsonschema
# import pandera as pa
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer

# internal
from configs.module.pandas_schema import SCHEMA

sys.path.append('.')

class DataLoader:

    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return pd.read_csv(data_config.path, delimiter=';')

    @staticmethod
    def validate_schema(data_point):
        """Data schema validation"""
        SCHEMA.validate(data_point)
        #jsonschema.validate({'data':data_point.tolist()},SCHEMA)

    @staticmethod
    def preprocess_data(dataset, test_size, random_state):
        """ Preprocess and splits into training and test"""
        return train_test_split(dataset, test_size=test_size, random_state=random_state)

    """
    # @staticmethod
    # def preprocess_pipeline(datapoint, ):

    #     #Loads and preprocess a datapoint with pipeline

    #     ## preprocessing pipeline
    #     numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    #     numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
    #                                         ('scaler', StandardScaler())])

    #     categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','y']
    #     categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #                                             ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    #     processed_data = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
    #                                                 ('cat', categorical_transformer, categorical_features)])

    #     return processed_data
    """

