import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import yaml
import joblib
import socket
import json
import numpy as np
import pandas as pd


## import model specific functions and variables
from src.models.train_model import model_train, model_load
from src.models.predict_model import model_predict, model_load
#from model import MODEL_VERSION, MODEL_VERSION_NOTE

## load config data
# folder to load config file
CONFIG_PATH = "conf/base"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name), 'r') as file:
        config = yaml.safe_load(file)

    return config

model_config = load_config("parameters.yml")

## load model parameters from conf/base/parameters.yml
MODEL_VERSION = model_config["model"]["version"]
MODEL_VERSION_NOTE = model_config["model"]["note"]

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/running', methods=['POST'])
def running():
    return render_template('running.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    basic predict function for the API
    """
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within")
        return jsonify([])

    if 'type' not in request.json:
        print("WARNING API (predict): received request, but no 'type' was found assuming 'numpy'")
        query_type = 'numpy'

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query
    query = request.json['query']
        
    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])

        
    ## load model
    #model = model_load(test=test)
    model = joblib.load(open('models/model-0-1.pkl', 'rb'))
    
    if not model:
        print("ERROR: model is not available")
        return jsonify([])

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features, model, test=test)

    _result = round(prediction[0], 2)

    print(output)

    #return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

    # _result = model_predict(query, model, test=test)
    result = {}
    
    ## convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    
    return(jsonify(result))


@app.route('/train', methods=['GET','POST'])
def train():
    """
    basic predict function for the API

    the 'mode' flag provides the ability to toggle between a test version and a 
    production verion of training
    """
    
    ## check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    print("... training model")
    model = model_train(test=test)
    print("... training complete")

    return(jsonify(True))
        
@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])
    
    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)

