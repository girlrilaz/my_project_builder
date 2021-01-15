ALA automation
==============================

Language assessment report generation automation.

## 1. Getting Started

To get started, first clone this repository to your machine. (see [docs] https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository for help on this)

## 2. Create virtual environment

Creating virtual environment to ensure reproducability of this project.

##### OPTION 1: If you have Anaconda distribution installed in your machine, set up a virtual environment with 'conda'

In your terminal or command line, the command to create virtual environment with conda :

<i> conda create --name [replace_with_env_name] python=[replace_with_python_version] </i>

for this project, copy and paste the command below
```
conda create --name ALA python=3.8
```
##### OPTION 2: Set up a Python virtual environment with 'python-dotenv'

Make sure you have python-dotenv installed

```
pip install python-dotenv
```

Then create a the virtual environment for isolating this project on your machine using the following command:

```
python -m venv .venv
```

## 3. Activate the virtual environment

##### OPTION 1: Conda virtual environment

Windows or Mac with Conda:

In the terminal or command line :

<i>conda activate [replace_with_env_name]</i>

```
conda activate ALA
```

##### OPTION 2: dotenv virtual environment

Windows:
```
.\.venv\Scripts\activate
```
Mac:
```
source .venv/bin/activate
```

## 3. Install Python dependencies

or pip:
```
pip install -r requirements.txt
```

If you need to extract the list of packages from this environment, 

```
pip freeze > requirements.txt
```

## Prepare the dataset and check the data processing report

```
python -m [something]
```

## Run the entire report generation

```
python -m [something]
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

### Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
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
    │   │   └── make_dataset.py  ## Change this
    │   │
    │   ├── reports         <- Scripts to run different report generation scripts
    │   │   │                 
    │   │   ├── predict_model.py ## Change this
    │   │   └── train_model.py ## Change this
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
<p><small>Modified by Nor Raymond with references to Quantum Black's open source Kedro framework - https://kedro.readthedocs.io/en/stable/ , https://github.com/quantumblacklabs/kedro</small></p>
