"""Make it easier to work with the dataset.

Create one .csv file that combines the train.csv and valid.csv, then augments the dataset. The
combined CSV appends the following columns to the existing dataset columns:

- Patient number
- Study number
- View number
- Age group (MeSH age group - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/)
- "Train" or "Validation" image

It also normalizes the labels to 0, 1, and -1 by converting floating point labels to integer (e.g.
0.0 to 0) and by filling in empty label columns with 0.

The code assumes that the dataset has been uncompressed into the same directory this file is in.

Usage: python3 -m preprocess > chexpert.csv

From another module: import this module and call the public function.
"""

import logging
import os
import re
import sys
import pandas as pd
import imagesize

# Names of the columns added with this code
COL_PATIENT_ID = 'Patient ID'
COL_STUDY_NUMBER = 'Study Number'
COL_VIEW_NUMBER = 'View Number'
COL_AGE_GROUP = 'Age Group'
COL_TRAIN_VALIDATION = 'Train/Validation'

# Values of columns added with this code
TRAIN = 'Train'
VALIDATION = 'Validation'

# Other useful constants
COL_LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
              'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('%(message)s'))
_logger = logging.getLogger(__name__)
_logger.addHandler(_ch)
_logger.setLevel(logging.INFO)


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


def _get_augmented_chexpert(add_image_size: bool = False) -> pd.DataFrame:
    """Augment the CheXpert dataset.

    Add columns described in the file header.

    Args:
        add_image_size (bool, optional): Add the image size (takes a few seconds). Defaults to
        False.

    Returns:
        pd.DataFrame: The dataset with the original and augmented columns.
    """
    chexpert_dir = _get_chexpert_directory()
    if not chexpert_dir:
        sys.exit('Cannot find the ChexPert directory')
    _logger.info('Found the dataset in %s', chexpert_dir)

    df = pd.concat(pd.read_csv(os.path.join(chexpert_dir, f)) for f in ['train.csv', 'valid.csv'])

    # Normalize the labels: replace empty ones with zero
    df.fillna(0, inplace=True)

    # Add the patient ID column by extracting it from the filename
    # Assume that the 'Path' column follows a well-defined format and extract from "patientNNNNN"
    _logger.info('Adding patient ID')
    df[COL_PATIENT_ID] = df.Path.apply(lambda x: int(x.split('/')[2][7:]))

    # Add the study number column, also assuming that the 'Path' column is well-defined
    _logger.info('Adding study number')
    df[COL_STUDY_NUMBER] = df.Path.apply(lambda x: int(x.split('/')[3][5:]))

    # Add the view number column, also assuming that the 'Path' column is well-defined
    _logger.info('Adding view number')
    view_regex = re.compile('/|_')
    df[COL_VIEW_NUMBER] = df.Path.apply(lambda x: int(re.split(view_regex, x)[4][4:]))

    # Add the MeSH age group column
    # Best reference I found for MeSH groups: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/
    # We have only complete years, so we can't use 'newborn'
    # Also prefix with zero because visualizers sort by ASCII code, not numeric value
    _logger.info('Adding age group')
    bins = [0, 2, 6, 13, 19, 45, 65, 80, 120]
    ages = ['(0-1) Infant', '(02-5) Preschool', '(06-12) Child', '(13-18) Adolescent',
            '(19-44) Adult', '(45-64) Middle age', '(65-79) Aged', '(80+) Aged 80']
    df[COL_AGE_GROUP] = pd.cut(df.Age, bins=bins, labels=ages, right=False)

    # Add the train/validation column
    _logger.info('Adding train/validation')
    df[COL_TRAIN_VALIDATION] = df.Path.apply(lambda x: TRAIN if 'train' in x else VALIDATION)

    # Add the image information column
    if add_image_size:
        logging.info('Adding image size (takes a few seconds)')
        size = [imagesize.get(f) for f in df.Path]
        df[['Width', 'Height']] = pd.DataFrame(size, index=df.index)

    # Optimize memory usage: use categorical values and small integer when possiblr
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html
    for c in ['Sex', 'Frontal/Lateral', 'AP/PA', COL_AGE_GROUP, COL_TRAIN_VALIDATION]:
        df[c] = df[c].astype('category')
    for c in ['Age', COL_PATIENT_ID, COL_STUDY_NUMBER, COL_VIEW_NUMBER] + COL_LABELS:
        df[c] = df[c].astype('int32')

    return df


def get_augmented_dataset(verbose: bool = False) -> pd.DataFrame:
    """Get an augmented version of the ChexPert dataset.

    See the module header for a description of the columns.

    Args:
        verbose (bool, optional): Turn verbose logging on/off. Defaults to off.

    Returns:
        pd.DataFrame: The dataset with the original and augmented columns.
    """
    _logger.setLevel(logging.INFO if verbose else logging.ERROR)
    return _get_augmented_chexpert()


def fix_dataset(df: pd.DataFrame):
    """Fix issues with the dataset (in place).

    See code for what is fixed.

    Args:
        df (pd.DataFrame): The dataset before it is fixed.
    """

    # The is one record with sex 'Unknown'. Change it to "Female" (it doesn't matter which sex we
    # pick because it is one record out of 200,000+).
    df.loc[df.Sex == 'Unknown', ['Sex']] = 'Female'


if __name__ == '__main__':
    chexpert = _get_augmented_chexpert(add_image_size=False)
    print(chexpert.to_csv(index=False))
