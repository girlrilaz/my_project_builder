# -*- coding: utf-8 -*-
import os
import glob
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Input and output path are saved in .env file. Please create one and
        save it under the project root directory.
    """
        
    logger = logging.getLogger(__name__)

    filename = os.getenv("FILENAME")
    logger.info(f"{bcolors.WARNING}Data Processing{bcolors.ENDC} : filename - '{filename}' from folder data/raw")

    # Read data from input path
    df = pd.read_csv(os.path.join(os.getenv("INPUT_FILEPATH"),os.getenv("FILENAME")))

    # TODO: process data here

    # Save data to output path
    df.to_csv(os.path.join(os.getenv("OUTPUT_FILEPATH"),os.getenv("FILENAME")), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
