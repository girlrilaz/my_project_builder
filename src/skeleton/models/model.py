# -*- coding: utf-8 -*-
"""Unet model"""

# standard library
import os, sys
import pandas as pd
import pickle
sys.path.append('.')

# internal
from .base_model import BaseModel
from utils.dataloader import DataLoader
from utils.logger import get_logger
from executor.model_trainer import ModelTrainer

# external
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

LOG = get_logger('change_to_model_name')

class ModelName(BaseModel):

    """Model Class"""

    def __init__(self, config):
        super().__init__(config)

        self.model = None
        self.dataset = None
        self.info = None
        self.model = None
        self.X_pipeline = []
        self.y_pipeline = []
        self.train_dataset = []
        self.test_dataset = []
        self.numerical = self.config.data.numerical_att
        self.categorical = self.config.data.categorical_att
        self.target = self.config.train.target_att
        self.test_size = self.config.train.test_size
        self.random_state = self.config.train.random_state
        self.model_name = self.config.model.name
        self.model_folder = self.config.model.folder
        self.model_version = self.config.model.version
        self.model_params = self.config.model.params

    def load_data(self):

        """Loads and Preprocess data """

        LOG.info(f'Loading {self.config.data.path} dataset .....' )

        self.dataset = DataLoader().load_data(self.config.data)

        LOG.info(f"..... validating all data")

        try:
            validate = DataLoader().validate_schema(self.dataset)
            if validate is None:
                LOG.info(f"PASS: Data validation passed.")
        except:
            LOG.error(f"FAIL: Data validation failed.")
            raise Exception("ERROR - FAIL:(dataloader) - invalid data schema") 
            # sys.exit(100) # exit if using log and no raise exception

        self.train_dataset, self.test_dataset = DataLoader().preprocess_data(self.dataset, self.test_size, self.random_state)

        self.X_train= DataLoader().feature_pipeline(self.numerical, self.categorical).fit(self.train_dataset).transform(self.train_dataset)
        self.y_train = DataLoader().target_pipeline(self.target).fit(self.train_dataset[self.target]).transform(self.train_dataset[self.target])

        self.X_test= DataLoader().feature_pipeline(self.numerical, self.categorical).fit(self.test_dataset).transform(self.test_dataset)
        self.y_test = DataLoader().target_pipeline(self.target).fit(self.test_dataset[self.target]).transform(self.test_dataset[self.target])

    def build(self):

        """
        Create the xgboost classifier with predefined initial parameters, user can overwright it by passing kw args in train
        """
        init_params = vars(self.model_params) #set in config
        self.model = XGBClassifier(**init_params, use_label_encoder=False)

        LOG.info('Model was built successfully')

    def train(self):

        """Compiles and trains the model with train dataset"""

        LOG.info('Training started')

        trainer = ModelTrainer(self.model, self.model_name, self.model_folder, self.model_version, 
                                self.X_train, self.y_train, vars(self.model_params))
        trainer.train()

    def evaluate(self): 

        """Predicts results for the test dataset"""

        LOG.info(f'Model predictions for test dataset')

        LOG.info(f"..... validating test data")
        
        ## schema checks
        try:
            validate = DataLoader().validate_schema(self.test_dataset)
            if validate is None:
                LOG.info(f"PASS: Test data validation passed.")
        except:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input schema.")

        ## input checks
        if isinstance(self.test_dataset,dict):
            self.test_dataset = pd.DataFrame(self.test_dataset)
        elif isinstance(self.test_dataset,pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {self.test_dataset} was given")

        ## features check
        test_features = sorted(self.test_dataset.columns.drop(['y']).tolist())
        data_features = sorted(self.dataset.columns.drop(['y']).tolist())
        if test_features != data_features:
            print(f"test features: {','.join(test_features)}")
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid features present") 
        
        model_pickle_name = self.model_name + '_' + self.model_folder + '.' + self.model_version + '.pickle'
        saved_model = os.path.join('models', 'saved_models', self.model_name, self.model_folder, model_pickle_name)  

        LOG.info(f"..... loading model {saved_model}")

        if not os.path.exists(saved_model):
            exc = (f"Model '{saved_model}' cannot be found. Did you train the full model?")
            raise Exception(exc)
    
        model = pickle.load(open(saved_model, 'rb'))

        ## make prediction and gather data for log entry
        y_pred = model.predict(self.X_test)

        LOG.info(f"..... starting model prediction on test data") 

        predictions = [round(value) for value in y_pred]

        LOG.info(f"..... model prediction completed") 

        # evaluate predictions using train_test split - quicker
        accuracy = accuracy_score(self.y_test, predictions)
        print("Train-Test split accuracy: %.2f%%" % (accuracy * 100.0))

        # evaluate predictions using Kfold method - good for sets model has not seen
        kfold = KFold(n_splits=10, random_state=7, shuffle = True)
        results = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
        print("K-fold validation accuracy (std): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

        # evaluate predictions using Stratified Kfold method - good for multiple classes or imbalanced dataset
        stratified_kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
        s_results = cross_val_score(model, self.X_test, self.y_test, cv=stratified_kfold)
        print("Stratified K-fold validation accuracy (std): %.2f%% (%.2f%%)" % (s_results.mean()*100, s_results.std()*100))

        return predictions


# if __name__ == '__main__':

#     m = ModelName(BaseModel)
#     m.load_data()