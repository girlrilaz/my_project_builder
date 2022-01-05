# -*- coding: utf-8 -*-
"""Data Loader"""

# internal
from configs.module.json_schema import SCHEMA

# external
import jsonschema

class DataLoader:

    """Data Loader class"""

    @staticmethod
    def load_data(data_config):

        """Loads dataset from path"""

        data=""
        return data
        #return tfds.load(data_config.path, with_info=data_config.load_with_info) # tensorflow dataset - local machine
        #return tfds.load(name=data_config.path, data_dir=data_config.bucket, with_info=data_config.load_with_info) # in cloud bucket

    # @staticmethod
    # def validate_schema(data_point):
    #     jsonschema.validate({'data':data_point.tolist()},SCHEMA)

    # @staticmethod
    # def preprocess_data(dataset):

    #     """ Preprocess and splits into training and test"""

    #     train = dataset['train'].map(lambda data: DataLoader._preprocess_train(data))
    #     test = dataset['test'].map(lambda data: DataLoader._preprocess_test(data))

    #     train_dataset = train
    #     test_dataset = test

    #     return train_dataset, test_dataset

    # @staticmethod
    # def _preprocess_train(datapoint):

    #     """ Loads and preprocess a single training datapoint """
        
    #     # example, normalizing datapoints
    #     for dp in datapoint:
    #         processed_data = DataLoader._normalize(dp)

    #     return processed_data

    # @staticmethod
    # def _preprocess_test(datapoint):

    #     """ Loads and preprocess a single test images """

    #     # example, normalizing datapoints
    #     for dp in datapoint:
    #         processed_data = DataLoader._normalize(dp)

    #     return processed_data

    # @staticmethod
    # def _normalize(single_datapoint):

    #     """ Normalise data"""
        
    #     normalized_data = ''

    #     return normalized_data


# TODO: DELETE LATER
if __name__ == '__main__':

    dload = DataLoader()