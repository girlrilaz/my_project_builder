# -*- coding: utf-8 -*-
""" main.py """

# standard libraries
import sys

# internal
from configs.module.config import CFG
from models.model import ModelName


def run():
    """Builds model, loads data, trains and evaluates"""

    if sys.argv[1] == "subset": 
        print(sys.argv[1])
        subset = True
    else:
        print(sys.argv[1])
        subset = False
    
    print(subset)

    model = ModelName(CFG)
    model.load_data(subset=subset)
    model.build()
    model.train(subset=subset)
    model.evaluate()


if __name__ == '__main__':
    run()