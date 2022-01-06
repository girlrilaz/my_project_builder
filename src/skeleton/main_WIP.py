# -*- coding: utf-8 -*-
""" main.py """

# internal
from configs.module.config import CFG
from models.model import ModelName


def run():
    """Builds model, loads data, trains and evaluates"""
    model = ModelName(CFG)
    model.load_data()
    # model.build()
    # model.train()
    # model.evaluate()


if __name__ == '__main__':
    run()