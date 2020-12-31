"""Explore the raw .csv files from the CheXpert dataset.

This is a tool to help understand the composition of the .csv files, e.g. how labels are used, in
its raw format, before massaging the data.

The main goal is to use it as a sanity check for the more complex code that manipulates the data,
e.g. creates tables and other views.
"""
import os
import pandas as pd
import chexpert_dataset as cd


def read_raw_csv(file: str) -> pd.DataFrame:
    """Read the .csv file in raw format, i.e. all labels are strings.

    The goal is to return the data as it is stored in the file, without any conversion, including
    conversions that Pandas may attempt to do if we don't force specific types for the columns.

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

df_train = read_raw_csv(os.path.join(directory, 'train.csv'))
df_valid = read_raw_csv(os.path.join(directory, 'valid.csv'))

# Count of labels per observation
df_train_counts = df_train[cd.OBSERVATION_ALL].apply(pd.Series.value_counts).T
df_valid_counts = df_valid[cd.OBSERVATION_ALL].apply(pd.Series.value_counts).T

train_labels = df_train_counts.columns.values
valid_labels = df_valid_counts.columns.values

print('\nCount of labels per observation - trainig set')
print('Labels used: {}'.format(train_labels))
print(df_train_counts)

print('\nCount of labels per observation - validation set')
print('Labels used: {}'.format(valid_labels))
print(df_valid_counts)

# Sanity check: verify that we found the labels we expected to find
# If this assert fails we either have a mistake in the code or the labels have changed
# If the labels have changed, we need to review all other places where we make assumptions about
# the label values
ALL_LABELS = ['', '-1.0', '0.0', '1.0']
POS_NEG_LABELS = ['0.0', '1.0']
assert train_labels.tolist() == ALL_LABELS
assert valid_labels.tolist() == POS_NEG_LABELS

# Sanity check: verify that we found the expected number of images in each set
# If this assert fails, the code is wrong or the number of images in the traninig/validation set
# has changed
# If the number of images has changed, we need to review all places where we make assumption about
# the number of images

# In the training set we expect to see all labels
assert df_train_counts.loc[cd.OBSERVATION_NO_FINDING][ALL_LABELS].sum() == cd.IMAGE_NUM_TRAINING

# In the validation set we expect to see only the positive and negative labels
assert df_valid_counts.loc[cd.OBSERVATION_NO_FINDING][POS_NEG_LABELS].sum() == \
    cd.IMAGE_NUM_VALIDATION
