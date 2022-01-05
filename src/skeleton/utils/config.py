# -*- coding: utf-8 -*-
"""Config class"""

# standard library
import json
import yaml

class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data):
        self.data = data
        # self.train = train
        # self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from configs/module"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data)
        #return cls(params.data, params.train, params.model) # if cfg has more than one params

    @classmethod
    def from_json_file(cls, cfg):
        """Creates config from configs/json"""
        with open('configs/json/config.json') as f:
            params = json.load(f, object_hook=HelperObject)
        return cls(params.data)

    @classmethod
    def from_yaml(cls, cfg):
        """Creates config from configs/yaml"""
        with open('configs/yaml/config.yml') as f:
            params = json.loads(json.dumps(yaml.safe_load(f)), object_hook=HelperObject)
        return cls(params.data)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)


# TODO: DELETE LATER
# if __name__ == '__main__':

#     # from_json
#     CFG = { "data": {
#                 "path": "",
#                 "bucket": ""} }
#     myJSON = json.loads(json.dumps(CFG), object_hook=HelperObject)
#     print(myJSON)

#     # from_json_file
#     with open('configs/json/config.json') as f:
#         data = json.load(f, object_hook=HelperObject)
#     print(data)

#     # # from yaml
#     with open('configs/yaml/config.yml') as f:
#         params2 = json.loads(json.dumps(yaml.safe_load(f)), object_hook=HelperObject)
#         print(params2)