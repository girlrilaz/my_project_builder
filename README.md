Blood Donation Prediction Project
==============================

Blood donation prediction project from Driven Data

## Getting Started

To get started, first clone this repository to your machine. (see [docs] https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository for help on this)

## Create and activate virtual environment

Creating virtual environment to ensure reproducability of this project.

### OPTION 1: Set up a Python virtual environment with Makefile in your CLI

1. open Makefile in the project root directory then under "GLOBAL", you will see ENV_NAME. Change the value to your desired environment name
2. In your command line, type 

```
make create_environment
```

### OPTION 2: Set up a Python virtual environment with 'conda'
```
conda create --name [replace_with_env_name] python=[replace_with_python_version]
```
### OPTION 3: Set up a Python virtual environment with 'python-dotenv'

Create a Python virtual environment for isolating this project on your machine using the following command:

```
python -m venv .venv
```

The current supported Python version is `3.7` so **please ensure that your virtual environment is created using this version of Python** to ensure compatability with the dev and prod environments.

Next, activate your virtual environment (see [docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment) for help on this).

Windows:
```
.\.venv\Scripts\activate
```
Mac:
```
source .venv/bin/activate
```
Windows or Mac with Conda:
```
conda activate [replace_with_env_name]
```

Finally, install the project dependencies using make command:

```
make requirements
```
or pip:
```
pip install -r requirements.txt
```
If you need to extract the list of packages from this environment, 

```
pip freeze > requirements.txt
```

## Prepare the dataset

```
make data
```

then prepare the training set features,

```
make features
```

## Run the application locally

TODO:

```
python -m run-model-train
```

## Unit Testing

The unit testing framework used for this project is [`unittest`](https://docs.python.org/3/library/unittest.html).
Tests are stored in the `blood-donation-prediction-3/unittests` directory.
An alternative unit testing framework that can be used - Pytest

### Testing locally with make command

To run individual unit testing, check the commands in Makefile. For example, to run the API test, enter the following in your command line
```
python -m unittests.ApiTests
```

To run all tests, 

```
python -m run-tests
```
or 
```
make test_all
```

### Testing locally with Coverage
This project template also uses [`coverage.py`](https://coverage.readthedocs.io/) for measuring test coverage.

To run the test suite locally, run the following command.
You do not need to run the app first.

```
coverage run -m unittest discover
```

You can then run the following command to generate a file-by-file coverage report:

```
coverage report -m
```

For a more detailed line-by-line coverage analysis, you can also generate a HTML coverage report:

```
coverage html
```

### Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
<p><small>Modified by Nor Raymond with references to Quantum Black's open source Kedro framework - https://kedro.readthedocs.io/en/stable/ , https://github.com/quantumblacklabs/kedro</small></p>
