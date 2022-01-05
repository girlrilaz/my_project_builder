#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from src.models.train_model import *
from src.models.predict_model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model = model_load(test=True)
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load(test=True)
    
        ## ensure that a list can be passed
        query = pd.DataFrame({'sepal_length': [5.1, 6.4],
                            'sepal_width': [3.5, 3.2],
                            'petal_length': [1.4, 4.5],
                            'petal_width': [0.2, 1.5]
        })

        result = model_predict(query, model, test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] in ['setosa','versicolor', 'virginica'])

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
