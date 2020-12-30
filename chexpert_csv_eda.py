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

# All (unique) labels seen across all observations in both files
all_labels = set()

for data in [df_train, df_valid]:
    for obs in cd.OBSERVATION_ALL:
        labels = data[obs].unique()
        for label in labels:
            all_labels.add(label)
        print('{}: {}'.format(obs, sorted(labels)))

all_labels = sorted(all_labels)
print('\nAll (unique) labels used in all files: {}'.format(sorted(all_labels)))

# Sanity check: verify that we found the labels we expected to find
# If this assert fails we either have a mistake in the code or the labels have changed
# If the labels have changed, we need to review all other places where we make assumptions about
# the label values
KNOWN_LABELS = ['', '-1.0', '0.0', '1.0']
assert all_labels == KNOWN_LABELS


def label_frequency(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    stats = pd.DataFrame(index=cd.OBSERVATION_ALL, columns=labels)
    for obs in cd.OBSERVATION_ALL:
        count = [len(df[df[obs] == x]) for x in labels]
        stats.loc[obs] = count
    # Sanity check: check a few columns for the number of images
    return stats


print('\nCount of labels per observation - trainig set')
count_train = label_frequency(df_train, all_labels)
print(count_train)

print('\nCount of labels per observation - validation set')
count_valid = label_frequency(df_valid, all_labels)
print(count_valid)

# Sanity check: verify that we found the expected number of images in each set
# If this assert fails, the code is wrong or the number of images in the traninig/validation set
# has changed
# If the number of images has changed, we need to review all places where we make assumption about
# the number of images

# In the training set we expect to see all labels
assert count_train.loc[cd.OBSERVATION_NO_FINDING][KNOWN_LABELS].sum() == cd.IMAGE_NUM_TRAINING

# In the validation set we expect to see only the positive and negative labels
assert count_valid.loc[cd.OBSERVATION_NO_FINDING][['1.0', '0.0']].sum() == cd.IMAGE_NUM_VALIDATION
