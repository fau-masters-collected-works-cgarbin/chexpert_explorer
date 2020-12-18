"""Make it easier to work with the dataset.

Create one .csv file that combines the train.csv and valid.csv, then augments the dataset. The
combined CSV appends the following columns to the existing dataset columns:

- Patient number
- Study number
- View number
- Age group (MeSH age group - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/)
- "Train" or "Test" image

It also normalizes the labels to 0, 1, and -1 by converting floating point labels to integer (e.g.
0.0 to 0) and by filling in empty label columns with 0.

The code assumes that the dataset has been uncompressed into the same directory this file is in.
"""

import logging
import os
import re
import sys
import pandas as pd

# Names of the columns added with this code
COL_PATIENT_ID = 'Patient ID'
COL_STUDY_NUMBER = 'Study Number'
COL_VIEW_NUMBER = 'View Number'
COL_AGE_GROUP = 'Age Group'
COL_TRAIN_TEST = 'Train/Test'

# Values of columns
TRAIN = 'Train'
TEST = 'Test'


def _get_chexpert_directory() -> str:
    """Determine the directory where the dataset is stored.

    There are two versions of the dataset, small and large. They are stored in CheXpert-v1.0-small
    and CheXpert-v1.0-large respectively. To make the code generic, this function finds out what
    version is installed.

    Note: assumes that 1) only one of the versions is installed and 2) that it is at the same level
    where this code is being executed.

    Returns:
        str: The name of the images directory or an empty string if it can't find one.
    """
    for entry in os.scandir('.'):
        if entry.is_dir() and re.match(r'CheXpert-v\d\.\d-', entry.name):
            return entry.name
    return ''


def _augment_chexpert() -> pd.DataFrame:
    """Augment the CheXpert dataset.

    Add columns described in the file header.

    Returns:
        pd.DataFrame: The dataset with the original and augmented columns.
    """
    chexpert_dir = _get_chexpert_directory()
    if not chexpert_dir:
        sys.exit('Cannot find the ChexPert directory')
    logging.info('Found the dataset in %s', chexpert_dir)

    df = pd.concat(pd.read_csv(os.path.join(chexpert_dir, f)) for f in ['train.csv', 'valid.csv'])

    # Add the patient ID column by extracting it from the filename
    # Assume that the 'Path' column follows a well-defined format and extract from "patientNNNNN"
    df[COL_PATIENT_ID] = df.Path.apply(lambda x: int(x.split('/')[2][7:]))

    # Add the study number column, also assuming that the 'Path' column is well-defined
    df[COL_STUDY_NUMBER] = df.Path.apply(lambda x: int(x.split('/')[3][5:]))

    # Add the view number column, also assuming that the 'Path' column is well-defined
    view_regex = re.compile('/|_')
    df[COL_VIEW_NUMBER] = df.Path.apply(lambda x: int(re.split(view_regex, x)[4][4:]))

    # Add the MeSH age group column
    # Best reference I found for MeSH groups: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/
    # We have only complete years, so we can't use 'newborn'
    # Also prefix with zero because visualizers sort by ASCII code, not numeric value
    bins = [0, 2, 6, 13, 19, 45, 65, 80, 120]
    ages = ['(0-1) Infant', '(02-5) Preschool', '(06-12) Child', '(13-18) Adolescent',
            '(19-44) Adult', '(45-64) Middle age', '(65-79) Aged', '(80+) Aged 80']
    df[COL_AGE_GROUP] = pd.cut(df.Age, bins=bins, labels=ages, right=False)

    # Add the train/test column
    df[COL_TRAIN_TEST] = df.Path.apply(lambda x: TRAIN if 'train' in x else TEST)

    print(df.head())
    print(df.tail())


logging.basicConfig(level=logging.INFO, format='%(message)s')

_augment_chexpert()
