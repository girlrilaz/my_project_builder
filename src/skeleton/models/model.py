# -*- coding: utf-8 -*-
"""XGboost model"""

# standard library
import sys
from .base_model import BaseModel
import pandas as pd
sys.path.append('.')

# external
from xgboost import XGBClassifier

# internal
from utils.logger import get_logger
from utils.dataloader import DataLoader
from executor.model_trainer import ModelTrainer
from executor.model_evaluator import ModelEvaluator

LOG = get_logger('xgboost')

class ModelName(BaseModel):

    """Model Class"""

    def __init__(self, config):
        super().__init__(config)

        self.model = None
        self.init_model = None
        self.dataset = None
        self.info = None
        self.model = None
        self.X_pipeline = []
        self.y_pipeline = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        # self.X = []
        # self.y = []
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
            LOG.critical("FAIL: data validation failed.")
            raise Exception("CRITICAL - FAIL:(dataloader) - invalid data schema")
            # sys.exit(100) # exit if using log and no raise exception

        # self.X, self.y = DataLoader().split_feature_target(self.dataset, self.target)
        # self.X_train, self.X_test, self.y_train ,self.y_test = DataLoader().preprocess_data(self.X, self.y, self.test_size, self.random_state)

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
        self.init_model = XGBClassifier(**init_params, use_label_encoder=False)

        LOG.info('Model was built successfully')

    def train(self):

        """Compiles and trains the model with train dataset"""

        trainer = ModelTrainer(self.init_model, self.model_name, self.model_folder, self.model_version,
                                self.X_train, self.y_train, vars(self.model_params),
                                self.numerical, self.categorical, self.target)
        trainer.train()

    def evaluate(self):

        """Predicts results for the test dataset"""

        LOG.info('Start evaluation on test dataset .....')

        LOG.info("..... validating test data")

        eval = ModelEvaluator(self.test_dataset, self.dataset, self.X_test, self.y_test)

        eval.test_data_validation()
        model = eval.model_load()
        predictions = eval.model_predict(model)
        eval.evaluation_report(model)

        return predictions
