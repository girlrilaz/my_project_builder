# -*- coding: utf-8 -*-

# standard library
import os

#internal
from utils.logger import get_logger

#external


LOG = get_logger('change_to_trainer_name')

class ModelTrainer:

    def __init__(self, model, input):
        self.model = model
        self.input = input
        self.train_log_dir = './logs/'
        self.model_save_path = './models/saved_models/'
        self.checkpoint_path = './checkpoints/'

    def train(self):

        '''
        Model Fitting and Training
        Save pickle models into saved_models
        '''

        LOG.info(f'Start model training ....')

        # TODO: model training here


        LOG.info(f"Saved checkpoint: {self.checkpoint_path}")

        save_path = os.path.join(self.model_save_path, "modelname/1/")

        # save model pickel here
        # TODO: save trained models here
        # tf.saved_model.save(self.model, save_path)

    def _write_summary(self, loss, epoch):

        '''
        Write training summary (console print and logs folder)
        '''
        print('')
        # with self.train_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=epoch)
        #     tf.summary.scalar('accuracy', self.metric.result(), step=epoch)
