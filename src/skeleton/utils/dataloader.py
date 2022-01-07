# -*- coding: utf-8 -*-
"""Data Loader"""

# standard library
import sys
import pandas as pd

# external
#import jsonschema
# import pandera as pa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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


    @staticmethod
    def feature_pipeline(numeric_features, categorical_features):

        """Loads and preprocess a datapoint with pipeline"""

    #         num_pipeline = Pipeline([
    #    ('selector', DataFrameSelector(num_attrs)),
    #    ('imputer', preprocessing.Imputer(strategy="median")),
    #    ('std_scaler', preprocessing.StandardScaler()),
    # ])

    # cat_pipeline = Pipeline([
    #     ('selector', DataFrameSelector(cat_attrs)),
    #     ('cat_enc', CategoricalEncoder(encoding="onehot-dense")),
    # ])
    
    # full_pipeline = FeatureUnion(transformer_list=[
    #     ('num_pipeline', num_pipeline),
    #     ('cat_pipeline', cat_pipeline),
    # ])

        ## preprocessing pipeline
        # numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                            ('scaler', StandardScaler())])

        # categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                                                

        X_pipeline = ColumnTransformer(transformers=[('num', numerical_pipeline, numeric_features),
                                                    ('cat', categorical_pipeline, categorical_features)])

        return X_pipeline

    @staticmethod
    def target_pipeline(numeric_features, categorical_features):

        #Loads and preprocess a datapoint with pipeline

        y_pipeline = ""

        return y_pipeline


