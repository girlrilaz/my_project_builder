# -*- coding: utf-8 -*-
"""Model Evaluator"""

# standard library
import os
import pickle
import pandas as pd

# external
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

# internal
# from utils.visualize_WIP import display
from utils.config import Config
from configs.module.config import CFG
from utils.dataloader import DataLoader
from utils.logger import get_logger
from utils.visualize_WIP import plot_cm, plot_pr_roc, plot_pr_vs_th

LOG = get_logger('xgboost_evaluator')

class ModelEvaluator:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.model_name = self.config.model.name
        self.model_folder = self.config.model.folder
        self.model_version = self.config.model.version
        # self.saved_path = '/Users/ninaraymond/documents/github/production_template_dl/Unet/saved_models/unet/1'

    def test_data_validation(self, test_dataset, dataset):

        ## schema checks
        try:
            validate = DataLoader().validate_schema(test_dataset)
            if validate is None:
                LOG.info("PASS: Test data validation passed.")
        except:
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid input schema.")

        ## input checks
        if isinstance(test_dataset,dict):
            test_dataset = pd.DataFrame(test_dataset)
        elif isinstance(test_dataset,pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {test_dataset} was given")

        ## features check
        test_features = sorted(test_dataset.columns.drop(['y']).tolist())
        data_features = sorted(dataset.columns.drop(['y']).tolist())
        if test_features != data_features:
            print(f"test features: {','.join(test_features)}")
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid features present")

    def model_load(self):

        model_pickle_name = self.model_name + '_' + self.model_folder + '.' + self.model_version + '.pickle'
        saved_model = os.path.join('models', 'saved_models', self.model_name, self.model_folder, model_pickle_name)
        LOG.info(f"..... loading model {saved_model}")

        if not os.path.exists(saved_model):
            exc = (f"model '{saved_model}' cannot be found. Did you train the full model?")
            raise Exception(exc)

        return pickle.load(open(saved_model, 'rb'))

    def model_predict(self, model, x_test, y_test):

        ## make prediction and gather data for log entry
        LOG.info("..... starting model prediction on test data")

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        predictions = [round(value) for value in y_pred]

        LOG.info("Model evaluation completed")

        # evaluate predictions using train_test split - quicker
        accuracy = accuracy_score(y_test, predictions)
        print(f"Train-Test split accuracy: {round(accuracy * 100.0,2)} %")

        # evaluate predictions using Kfold method - good for sets model has not seen
        kfold = KFold(n_splits=10, random_state=7, shuffle = True)
        results = cross_val_score(model, x_test, y_test, cv=kfold)
        print(f"K-fold validation accuracy (std): {round(results.mean()*100,2)} % ({round(results.std()*100,2)} %)")

        # evaluate predictions using Stratified Kfold method - good for multiple classes or imbalanced dataset
        stratified_kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
        s_results = cross_val_score(model, x_test, y_test, cv=stratified_kfold)
        print(f"Stratified K-fold validation accuracy (std): {round(s_results.mean()*100,2)} % ({round(s_results.std()*100,2)} %)")

        predictions = {'y_pred':y_pred,'y_proba':y_proba}

        return predictions

    def evaluation_report(self, y_act, y_pred, y_proba, title="", cmap="Blues"):

        """
        create the classification reports with confusion matrix
        arguement:
        y_act -- Actual label of the class on the test data.
        y_pred -- Prediction by model on the test data.
        y_proba -- Probabilities as predicted by model on the test data.
        """
        plot_pr_roc(y_act, y_proba, "", "darkorange", True, title)  
        plot_cm(y_act, y_pred, title + " Confusion Matrix", cmap) 
        print("\n\n Classification Report ", title, "\n\n")  
        print(classification_report(y_act, y_pred))