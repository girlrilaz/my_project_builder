
import pickle
import pandas as pd
import sys
sys.path.append('.')

# from utils.visualize import display
from utils.config import Config
from configs.module.config import CFG

class ModelInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.saved_path = './models/saved_models/XGBoost/1/XGBoost_1.0.0.pickle'
        self.model = pickle.load(open(self.saved_path, 'rb'))
        # self.predict = self.model.signatures["serving_default"]
        # print(self.predict.structured_outputs)

    def preprocess(self, query):

        ## input checks
        if isinstance(query, dict):
            query_df = pd.DataFrame(query)
        elif isinstance(query, pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {query.dtype()} was given")
        return query_df

    def infer(self, query):
  
        query = self.preprocess(query)
        pred = self.model.predict(query)
        # pred = pred.numpy().tolist()
        return {'prediction_output':pred}


if __name__ == '__main__':

    query = {'age': [30],
            'job': ["unemployed"],
            "marital": ["married"],
            "education": ["primary"],
            "default": ["no"],
            "balance": [1787],
            "housing": ["no"],
            "loan": ["no"],
            "contact": ["cellular"],
            "day": [19],
            "month": ["oct"],
            "duration": [79],
            "campaign": [1],
            "pdays": [-1],
            "previous": [0],
            "poutcome":["unknown"]
    }
    print(ModelInferrer().infer(query))

    # print(ModelInferrer().infer(query))
