# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "source" : "local",
        "type" : "csv",
        "path": "data/raw/bank.csv",
        "bucket": "",
        "numerical": ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
        "categorical" : ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome","y"]
    },
    "train": {
        "target_col": "y",
        "test_size": 0.33,
        "random_state": 123,
        "metrics": ["accuracy"]
    },
}