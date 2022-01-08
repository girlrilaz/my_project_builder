# -*- coding: utf-8 -*-
"""XGboost model"""

# standard library
import os
import sys
from .base_model import BaseModel
import pickle
import pandas as pd
sys.path.append('.')

# external
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

# internal
from utils.logger import get_logger
from utils.dataloader import DataLoader
from executor.model_trainer import ModelTrainer

LOG = get_logger('xgboost')

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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
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

        LOG.info(f'loading {self.config.data.path} dataset .....' )

        self.dataset = DataLoader().load_data(self.config.data)

        LOG.info("..... validating all data")

        try:
            validate = DataLoader().validate_schema(self.dataset)
            if validate is None:
                LOG.info("PASS: data validation passed.")
        except:
            LOG.error("FAIL: data validation failed.")
            raise Exception("ERROR - FAIL:(dataloader) - invalid data schema")
            # sys.exit(100) # exit if using log and no raise exception

        self.train_dataset, self.test_dataset = DataLoader().preprocess_data(self.dataset, self.test_size, self.random_state)

        self.X_train= DataLoader().feature_pipeline(self.numerical, self.categorical) \
            .fit(self.train_dataset).transform(self.train_dataset)
        self.y_train = DataLoader().target_pipeline(self.target).fit(self.train_dataset[self.target]) \
            .transform(self.train_dataset[self.target])

        self.X_test= DataLoader().feature_pipeline(self.numerical, self.categorical).fit(self.test_dataset) \
            .transform(self.test_dataset)
        self.y_test = DataLoader().target_pipeline(self.target).fit(self.test_dataset[self.target]) \
            .transform(self.test_dataset[self.target])

    def build(self):

        """
        Create the xgboost classifier with predefined initial parameters, user can overwright it by passing kw args in train
        """
        init_params = vars(self.model_params) #set in config
        self.model = XGBClassifier(**init_params, use_label_encoder=False)

        LOG.info('Model was built successfully')

    def train(self):

        """Compiles and trains the model with train dataset"""

        trainer = ModelTrainer(self.model, self.model_name, self.model_folder, self.model_version,
                                self.X_train, self.y_train, vars(self.model_params))
        trainer.train()

    def evaluate(self):

        """Predicts results for the test dataset"""

        LOG.info('Start evaluation on test dataset .....')

        LOG.info("..... validating test data")

        ## schema checks
        try:
            validate = DataLoader().validate_schema(self.test_dataset)
            if validate is None:
                LOG.info("PASS: Test data validation passed.")
        except:
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid input schema.")

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
            exc = (f"model '{saved_model}' cannot be found. Did you train the full model?")
            raise Exception(exc)

        model = pickle.load(open(saved_model, 'rb'))

        ## make prediction and gather data for log entry
        y_pred = model.predict(self.X_test)

        LOG.info("..... starting model prediction on test data")

        predictions = [round(value) for value in y_pred]

        LOG.info("Model evaluation completed")

        # evaluate predictions using train_test split - quicker
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Train-Test split accuracy: {round(accuracy * 100.0,2)} %")

        # evaluate predictions using Kfold method - good for sets model has not seen
        kfold = KFold(n_splits=10, random_state=7, shuffle = True)
        results = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
        print(f"K-fold validation accuracy (std): {round(results.mean()*100,2)} % ({round(results.std()*100,2)} %)")

        # evaluate predictions using Stratified Kfold method - good for multiple classes or imbalanced dataset
        stratified_kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
        s_results = cross_val_score(model, self.X_test, self.y_test, cv=stratified_kfold)
        print(f"Stratified K-fold validation accuracy (std): {round(s_results.mean()*100,2)} % ({round(s_results.std()*100,2)} %)")

        return predictions