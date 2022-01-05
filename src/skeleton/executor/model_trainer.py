# -*- coding: utf-8 -*-

# standard library
import os

#internal
from utils.logger import get_logger

#external
import tensorflow as tf

LOG = get_logger('trainer')

class ModelTrainer:

    def __init__(self, model, input, loss_fn, optimizer, metric, epoches):
        self.model = model
        self.input = input

        self.train_log_dir = './logs/'
        # self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.model_save_path = './models/saved_models/'

    def train(self):

        '''
        Model Fitting and Training
        Save pickle models into saved_models
        '''

        LOG.info(f'Start model training ....')

        save_path = self.checkpoint_manager.save()

        LOG.info("Saved checkpoint: {}".format(save_path))

        save_path = os.path.join(self.model_save_path, "modelname/1/")

        # save model pickel here
        # tf.saved_model.save(self.model, save_path)

    def _write_summary(self, loss, epoch):

        '''
        Write training summary (console print and logs folder)
        '''
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.scalar('accuracy', self.metric.result(), step=epoch)
