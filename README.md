My Project Builder
==============================

Building Python packages

## 1. Store projects in a specific folder in src

## 2. Configure the setup.py file to point to the project that you want to package

## 3. Build the package

Make sure you have pep517 installed

```
pip install pep517
```

To build the distribution package (make sure the setup.py was configured correctly) :

```
make dist
```
Check in the 'dist' folder that there are two created files, .whl and .tar.

## 4. Install the package distribution to your environment

Activate your environment where you want the package to be installed to

```
conda activate [env_name]
```

Then install using pip

```
pip install dist/[package_name].whl
```

To check if the package has been installed into your environment, navigate to :

/opt/anaconda3/envs/py38/lib/[python_version]/site-packages

then check if your package name exists.

*python_version is the python version installed in your environment for example python3.8

## To upload package to a repository

We can upload a package to a repository such as PyPi, but for testing purposes, best to use a testing platform such as testpypi https://test.pypi.org/

Create an account and remember your username and password.

To upload your package to testpypi, first install twine 

```
pip install twine
```

Then 

```
python -m twine upload --repository testpypi dist/*
```

You will be prompted to enter your testpypi username and password which was obtained when you created an account

Please ensure that your package name is unique, otherwise the upload will fail


## To remove and clean all build dist associated files

```
make clean-build
```

## To rebuild a different package

clean the previous builds
```
make clean-build
```
Change the configurations in setup.py and then 

```
make dist
```

## Unit Testing

The unit testing framework used for this project is [`unittest`](https://docs.python.org/3/library/unittest.html).
Tests are stored in the `tests` directory.
An alternative unit testing framework that can be used - Pytest

### Testing locally with make command

To run individual unit testing, check the commands in Makefile. For example, to run the API test, enter the following in your command line
```
python -m tests.[testname]
```

To run all tests, 

```
python -m tests
```

or 

```
make test
```

### Project Organization

------------

```
.
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── src
└── tests

```
--------

