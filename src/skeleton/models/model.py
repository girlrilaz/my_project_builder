# -*- coding: utf-8 -*-
"""Unet model"""

# standard library

# internal
from .base_model import BaseModel
from utils.dataloader import DataLoader
from utils.logger_WIP import get_logger
from executor.model_trainer import ModelTrainer

# external

LOG = get_logger('unet')
# print(f'Tensorflow version: {tf.__version__}')
# tf.config.set_visible_devices([], 'GPU')

class ModelName(BaseModel):

    """Model Class"""

    def __init__(self, config):
        super().__init__(config)

        self.model = None
        # self.output_channels = self.config.model.output
        # self.dataset = None
        # self.info = None
        # self.train_dataset = []
        # self.test_dataset = []

    def load_data(self):

        """Loads and Preprocess data """

        LOG.info(f'Loading {self.config.data.path} dataset...' )

        self.dataset, self.info = DataLoader().load_data(self.config.data)
        self.train_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.batch_size, self.buffer_size, self.image_size)

    def build(self):

        """ Builds the model based """

        self.model = ""

        LOG.info('Model was built successfully')

    def train(self):

        """Compiles and trains the model with train dataset"""

        LOG.info('Training started')

        trainer = ModelTrainer(self.model, self.train_dataset)
        trainer.train()

    def evaluate(self):

        """Predicts results for the test dataset"""

        predictions = []
        LOG.info(f'Model predictions for test dataset')

        for im in self.test_dataset.as_numpy_iterator():
            DataLoader().validate_schema(im[0])
            break

        for predicted in self.test_dataset:
            LOG.info(f'Predicting segmentation map {predicted}')
            predictions.append(self.model.predict(predicted))
            
        return predictions