# -*- coding: utf-8 -*-
"""Unet model"""

# standard library
import sys
sys.path.append('.')

# internal
from .base_model import BaseModel
from utils.dataloader import DataLoader
# from utils.logger_WIP import get_logger
# from executor.model_trainer import ModelTrainer

# external

# LOG = get_logger('unet')

class ModelName(BaseModel):

    """Model Class"""

    def __init__(self, config):
        super().__init__(config)

        # self.model = None
        # self.output_channels = self.config.model.output
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
        self.model_params = self.config.model.params

    def load_data(self):

        """Loads and Preprocess data """

        # LOG.info(f'Loading {self.config.data.path} dataset...' )
        print(f'Loading {self.config.data.path} dataset...' )

        self.dataset = DataLoader().load_data(self.config.data)
        self.train_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.test_size, self.random_state)

        # TODO: uncomment when ready to use
        # self.X_train= DataLoader().feature_pipeline(self.numerical, self.categorical).fit(self.train_dataset).transform(self.train_dataset)
        # self.y_train = DataLoader().target_pipeline(self.target).fit(self.train_dataset[self.target]).transform(self.train_dataset[self.target])

    def build(self):

        """
        Create the xgboost classifier with predefined parameters, user can overwright it by passing kw args
        """

        print(self.model_params)

        # param['scale_pos_weight']= 1

        # param = {}
        # if self.model_params:
        #     for k, v in self.model_params.items():
        #         param[k] = self.model_params[k]
        
        # print(param)
                    
        # self.model = XGBClassifier(**param)

        # # LOG.info('Model was built successfully')
        print('Model was built successfully')

    # def train(self):

    #     """Compiles and trains the model with train dataset"""

    #     LOG.info('Training started')

    #     trainer = ModelTrainer(self.model, self.train_dataset)
    #     trainer.train()

    # def evaluate(self):

    #     """Predicts results for the test dataset"""

    #     predictions = []
    #     LOG.info(f'Model predictions for test dataset')

    #     for im in self.test_dataset.as_numpy_iterator():
    #         DataLoader().validate_schema(im[0])
    #         break

    #     for predicted in self.test_dataset:
    #         LOG.info(f'Predicting segmentation map {predicted}')
    #         predictions.append(self.model.predict(predicted))
            
    #     return predictions


# if __name__ == '__main__':

#     m = ModelName(BaseModel)
#     m.load_data()