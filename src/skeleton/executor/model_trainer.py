# -*- coding: utf-8 -*-

# standard library
import os, re
import pickle

from xgboost.sklearn import XGBClassifier

#internal
from utils.logger import get_logger

#external
from sklearn.model_selection import GridSearchCV, StratifiedKFold


LOG = get_logger('change_to_trainer_name')

class ModelTrainer:

    def __init__(self, model, X_train, y_train, init_params):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.init_params = init_params
        self.train_log_dir = './logs/'
        self.model_save_path = './models/saved_models/'
        self.checkpoint_path = './checkpoints/'

    def train(self):

        '''
        Model Fitting and Training
        Save pickle models into saved_models
        '''

        LOG.info(f'Start model training ....')

        LOG.info(f'.... grid searchingå')

        grid_params =  {
            "nthread":[4],
            "booster": ["gbtree"], 
            "n_estimators": [20, 40],
            "objective": ["binary:logistic"], 
            "learning_rate" : [0.25, 0.5],
            "eval_metric": ["error"], 
            "eta": [0.3], 
            "gamma": [0], 
            "max_depth": [6], 
            "min_child_weight": [4], 
            "max_delta_step": [0], 
            "subsample": [1], 
            "colsample_bytree": [1], 
            "seed": [0], 
            "scale_pos_weight": [1]
        } 
    
        #grid = GridSearchCV(self.model, param_grid=self.init_params, cv=10, n_jobs=1)
        grid =  GridSearchCV(self.model, param_grid=grid_params, n_jobs=5, 
                   cv=StratifiedKFold(n_splits=10, random_state=0, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

        grid.fit(self.X_train, self.y_train)

        best_params = grid.best_params_
        best_params = {re.sub("clf__","",key):value for key,value in best_params.items()}

        ## fit model on training data
        final_model = XGBClassifier(**best_params, use_label_encoder=False)
        final_model.fit(self.X_train,self.y_train)

        print(self.model)
        print(final_model)

        # LOG.info(f"Saved checkpoint: {self.checkpoint_path}")

        # save model pickel here
        save_path = os.path.join(self.model_save_path, "modelname/1/")
        os.makedirs(save_path, exist_ok = True) 
        pickle.dump(final_model, open(os.path.join(save_path, 'model.pickle'),'wb'))

        LOG.info(f"Saved model: {save_path}")

    def _write_summary(self, loss, epoch):

        '''
        Write training summary (console print and logs folder)
        '''
        print('')
        # with self.train_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=epoch)
        #     tf.summary.scalar('accuracy', self.metric.result(), step=epoch)



## MODEL PREDICT / EVALUATION

    # def create_y_weights(self, w, y):
    #     """
    #     helper routine to create  weight of size y .All the values are same i.e w
    #     """
    #     r = y.copy()
    #     r[r == 1] = w  
    #     r[r == 0] = 1
    #     return r 

    # def model_cross_validation(self, name_model, X, y, weight=1, fold=3):
    #     """
    #     utility function to find the best model using k fold cross validation
        
    #     arguements: 
    #     name_model -- name model
    #     X -- transformed feature dataset
    #     y -- corresponding target dataset
    #     weight -- weight to apply to positive sample. Default to 1 and we will see how it can be used with different weights.
        
    #     return: cross validated model with k == 3. It will return prob as well as prediction
    #     """
        
    #     # get the weights of w of size y
    #     y_weights = self.create_y_weights(weight, y)
        
    #     # get prediction
    #     y_proba = cross_val_predict(
    #                 name_model, X, y, cv=fold, method="predict_proba",
    #                 fit_params={'eval_metric':'auc',
    #                         'sample_weight':y_weights})
        
    #     # get probabilies
    #     y_pred = cross_val_predict(
    #                 name_model, X, y, cv=fold, method="predict",
    #                 fit_params={'eval_metric':'auc',
    #                         'sample_weight':y_weights})
        
    #     # result
    #     cv_result = {
    #         'model': name_model,
    #         'X':X,
    #         'y':y,
    #         'weight': weight,
    #         'proba':y_proba,
    #         'pred':y_pred,
    #     }
        
    #     return cv_result

    # def model_pred(self, X_train, y_train):

    #     xgb_model = self.model

    #     # use cross validation to find best model
    #     xgb_predict_obj = self.model_cross_validation(xgb_model, X_train, y_train, weight=1)

    #     xgb_cls_1_proba = xgb_predict_obj['proba'][:,1]
    #     xgb_y_pred = xgb_predict_obj['pred']

