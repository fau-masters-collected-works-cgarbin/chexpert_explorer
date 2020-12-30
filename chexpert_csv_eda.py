"""Explore the raw .csv files from the CheXpert dataset.

This is a tool to help understand the composition of the .csv files, e.g. how labels are used.
"""
import os
import pandas as pd
import chexpert_dataset as cd


def read_raw_csv(file: str) -> pd.DataFrame:
    """Read the .csv file in raw format, i.e. all labels are strings.

    Args:
        file (str): full path to the file

    Returns:
        pd.DataFrame: A DataFrame with the file contents
    """

    # Read all observations columns as string to preserve their contents
    obs_dtype = {}
    for obs in cd.OBSERVATION_ALL:
        obs_dtype[obs] = 'str'

    # Suppress NaN determination to read the raw values from the observations
    df = pd.read_csv(file, dtype=obs_dtype, keep_default_na=False)

    return df


directory = cd.CheXpert.find_directory()
if not directory:
    raise RuntimeError('Cannot find the CheXpert directory')

for file in ['train.csv', 'valid.csv']:
    print('\nUnique values for {}'.format(file))
    df = read_raw_csv(os.path.join(directory, file))
    for obs in cd.OBSERVATION_ALL:
        print('{}: {}'.format(obs, sorted(df[obs].unique())))
