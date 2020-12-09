import os
import logging
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv

def load_data():

    data_dir = os.path.join(os.getenv("OUTPUT_FILEPATH"),os.getenv("FILENAME"))
    df = pd.read_csv(data_dir)
       
    ## pull out the target and remove uneeded columns
    _y = df.pop('species')
    y = np.zeros(_y.size)
    y[_y==0] = 1 
    df.head()
    X = df

    return(X, y)

if __name__ == "__main__":
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    print(load_data())