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
from utils.visualize import plot_cm, plot_pr_roc, plot_pr_vs_th

LOG = get_logger('xgboost_evaluator')

class ModelEvaluator:
    def __init__(self, test_dataset, dataset, X_test, y_test):
        self.config = Config.from_json(CFG)
        self.test_dataset = test_dataset
        self.dataset = dataset
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = self.config.model.name
        self.model_folder = self.config.model.folder
        self.model_version = self.config.model.version
        self.predictions = {}
        self.report_path = './evaluation/report'
        self.plots_path = './evaluation/plots'
        # self.saved_path = '/Users/ninaraymond/documents/github/production_template_dl/Unet/saved_models/unet/1'

    def test_data_validation(self):

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

    def model_load(self):

        model_pickle_name = self.model_name + '_' + self.model_folder + '.' + self.model_version + '.pickle'
        saved_model = os.path.join('models', 'saved_models', self.model_name, self.model_folder, model_pickle_name)
        LOG.info(f"..... loading model {saved_model}")

        if not os.path.exists(saved_model):
            exc = (f"model '{saved_model}' cannot be found. Did you train the full model?")
            raise Exception(exc)

        return pickle.load(open(saved_model, 'rb'))

    def model_predict(self, model):

        ## make prediction and gather data for log entry
        LOG.info("..... starting model prediction on test data")

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        self.predictions = [round(value) for value in y_pred]

        LOG.info('Model evaluation completed')

        self.predictions = {'y_pred':y_pred,'y_proba':y_proba}

        return self.predictions

    def evaluation_report(self, model):  #, y_proba, title="", cmap="Blues"

        model_name_version = self.model_name + '_' + self.model_folder + '.' + self.model_version
        os.makedirs(self.report_path, exist_ok = True)

        print("\n Accuracy Report - ", model_name_version, "")

        ## GENERATE METRIC REPORTS
        # evaluate predictions using train_test split - quicker
        accuracy = accuracy_score(self.y_test, self.predictions['y_pred'])
        acc_1 = f"\n Train-Test split accuracy: {round(accuracy * 100.0,2)} %"
        print(acc_1)

        # evaluate predictions using Kfold method - good for sets model has not seen
        kfold = KFold(n_splits=10, random_state=7, shuffle = True)
        results = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
        acc_2 = f" K-fold validation accuracy (std): {round(results.mean()*100,2)} % ({round(results.std()*100,2)} %)"
        print(acc_2)

        # evaluate predictions using Stratified Kfold method - good for multiple classes or imbalanced dataset
        stratified_kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
        s_results = cross_val_score(model, self.X_test, self.y_test, cv=stratified_kfold)
        acc_3 = f" Stratified K-fold validation accuracy (std): {round(s_results.mean()*100,2)} % ({round(s_results.std()*100,2)} %)"
        print(acc_3)

        # evaluate predictions using confusion matrix
        print("\n Classification Report - ", model_name_version, "\n\n")
        print(classification_report(self.y_test, self.predictions['y_pred']))

        # ## GENERATE PLOTS
        os.makedirs(self.plots_path, exist_ok = True)
        plot_pr_roc(self.y_test, self.predictions['y_proba'][:,1], self.plots_path, "", "darkorange", False, model_name_version)
        plot_cm(self.y_test, self.predictions['y_pred'], self.plots_path, model_name_version + " - Confusion Matrix")

        LOG.info("Model evaluation completed")
        LOG.info(f"Evaluation reports saved in : {self.report_path}")
        LOG.info(f"Evaluation plots saved in : {self.plots_path}")
 
