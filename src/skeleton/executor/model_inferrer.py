
import pickle
import pandas as pd
import sys
sys.path.append('.')

# from utils.visualize import display
from utils.config import Config
from configs.module.config import CFG
from utils.dataloader import DataLoader

class ModelInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.numerical = self.config.data.numerical_att
        self.categorical = self.config.data.categorical_att
        self.target = self.config.train.target_att
        self.saved_path = './models/saved_models/XGBoost/1/XGBoost_1.0.0.pickle'
        self.model = pickle.load(open(self.saved_path, 'rb'))
        # self.predict = self.model.signatures["serving_default"]
        # print(self.predict.structured_outputs)

    def preprocess(self, query):

        ## input checks
        if isinstance(query, dict):
            query = pd.DataFrame(query)
        elif isinstance(query, pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {type(query)} was given")

        query_df = DataLoader().feature_pipeline(self.numerical, self.categorical).fit(query).transform(query)

        return query_df

    def infer(self, query):
  
        query = self.preprocess(query)

        pred = self.model.predict(query)
        pred = pred.tolist()
        return {'prediction_output':pred}


if __name__ == '__main__':

    # query = {'age': [30, 33],
    #         'job': ["unemployed", "services"],
    #         "marital": ["married", "married"],
    #         "education": ["primary", "secondary"],
    #         "default": ["no", "no"],
    #         "balance": [1787, 4789],
    #         "housing": ["no", "yes"],
    #         "loan": ["no", "yes"],
    #         "contact": ["cellular", "cellular"],
    #         "day": [19, 11],
    #         "month": ["oct", "may"],
    #         "duration": [79, 220],
    #         "campaign": [1, 1],
    #         "pdays": [-1, 339],
    #         "previous": [0, 4],
    #         "poutcome":["unknown", "failure"]
    # }

    df = pd.read_csv("./data/raw/bank.csv", delimiter=';')
    query = df.drop(['y'], axis=1)

    print(ModelInferrer().infer(query))
