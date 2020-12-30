"""CheXpert LaTex exporter.

Export CheXpert statistics and graphs to be imported in LaTex documents.
"""

import os
import re
from numpy.core.defchararray import center
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chexpert_dataset as cd

sns.set_style("whitegrid")
sns.set_palette("Set2", 4, 0.75)

# Destination directories, with path separator at the end to simplify the code
DIR_TABLES = os.path.join('..', 'chexpert-datasheet', 'tables') + os.sep
DIR_GRAPHS = os.path.join('..', 'chexpert-datasheet', 'graphs') + os.sep

IMAGES = 'Images'
PATIENTS = 'Patients'
FLOAT_FORMAT = '{:0,.1f}'.format
INT_FORMAT = '{:,}'.format

SHORT_LABELS = {'Enlarged Cardiomediastinum': 'Enlarged Card.'}

chexpert = cd.CheXpert()
chexpert.fix_dataset()
# Make code a bit simpler
df = chexpert.df

# Count of patients and images in the training and validation datasets
name = 'patient-images-train-validate'
caption = 'Number of patients and images'
stats = df.groupby([cd.COL_TRAIN_VALIDATION], as_index=True, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
assert stats.loc[cd.TRAINING][PATIENTS].sum() == cd.PATIENT_NUM_TRAINING
assert stats[PATIENTS].sum() == cd.PATIENT_NUM_TOTAL
assert stats[IMAGES].sum() == cd.IMAGE_NUM_TOTAL
stats.to_latex(buf=DIR_TABLES+name+'.tex',
               formatters=[INT_FORMAT] * stats.shape[1],
               float_format=FLOAT_FORMAT, index_names=False,
               caption=caption, label='tab:' + name, position='h!')

# Summary statistic of images per patient
name = 'patient-images-stats-summary'
caption = 'Summary statistics for images per patient'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID], as_index=True, observed=True).agg(
    Images=(cd.COL_VIEW_NUMBER, 'count'))
assert stats.loc[cd.TRAINING][IMAGES].sum() == cd.IMAGE_NUM_TRAINING
assert stats[IMAGES].sum() == cd.IMAGE_NUM_TOTAL
summary = stats.groupby([cd.COL_TRAIN_VALIDATION], as_index=True).agg(
    Min=(IMAGES, 'min'), Max=(IMAGES, 'max'), Median=(IMAGES, 'median'), Mean=(IMAGES, 'mean'),
    Std=(IMAGES, 'std'))
summary.to_latex(buf=DIR_TABLES+name+'.tex',
                 float_format=FLOAT_FORMAT, index_names=False,
                 caption=caption, label='tab:' + name, position='h!')

# Binned number of images per patient (continuing from above, where the number of images was added)
name = 'patient-images-stats-distribution'
caption = 'Distribution of images per patient'
bins = [0, 1, 2, 3, 10, 100]
bin_labels = ['1 image', '2 images', '3 images', '4 to 10 images', 'More than 10 images']
IMAGE_SUMMARY = 'Number of images'
stats[IMAGE_SUMMARY] = pd.cut(stats.Images, bins=bins, labels=bin_labels, right=True)

summary = stats.reset_index().groupby([IMAGE_SUMMARY], as_index=True, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique))
assert summary[PATIENTS].sum() == cd.PATIENT_NUM_TOTAL
summary.to_latex(buf=DIR_TABLES+name+'.tex',
                 formatters=[INT_FORMAT] * summary.shape[1],
                 float_format=FLOAT_FORMAT, index_names=True,
                 caption=caption, label='tab:' + name, position='h!')

# Frequency of labels in the training and validation sets


def label_image_frequency(df: pd.DataFrame) -> pd.DataFrame:
    observations = cd.OBSERVATION_OTHER + cd.OBSERVATION_PATHOLOGY
    images_in_set = len(df[cd.COL_VIEW_NUMBER])
    ALL_LABELS = [cd.LABEL_POSITIVE, cd.LABEL_NEGATIVE, cd.LABEL_UNCERTAIN, cd.LABEL_NO_MENTION]
    COL_NAMES = ['Pos', '%', 'Neg', '%', 'Unc', '%', 'No mention', '%']
    stats = pd.DataFrame(index=observations, columns=COL_NAMES)
    for obs in observations:
        count = [len(df[df[obs] == x]) for x in ALL_LABELS]
        pct = [c*100/images_in_set for c in count]
        stats.loc[obs] = [x for t in zip(count, pct) for x in t]
    # Sanity check: check a few columns for the number of images
    cols_no_pct = [v for v in COL_NAMES if v != '%']
    assert stats.loc[cd.OBSERVATION_NO_FINDING][cols_no_pct].sum() == images_in_set
    assert stats.loc[cd.OBSERVATION_PATHOLOGY[1]][cols_no_pct].sum() == images_in_set
    return stats


def generate_image_frequency_table(df: pd.DataFrame, name: str, caption: str, file: str,
                                   pos_neg_only: bool = False):
    stats = label_image_frequency(df)
    stats.rename(index=SHORT_LABELS, inplace=True)
    if pos_neg_only:
        # Assume pos/neg count and % are the first columns
        stats = stats.iloc[:, :4]

    table = stats.to_latex(column_format='l' + 'r' * stats.shape[1],
                           formatters=[INT_FORMAT, '{:.1%}'.format] * (stats.shape[1]//2),
                           float_format=FLOAT_FORMAT, index_names=True,
                           caption=caption, label='tab:' + name, position='h!')
    # Change font size (no option for that in to_latex)
    table = table.replace('\\centering', '\\scriptsize\n\\centering')
    with open(file, 'w') as f:
        print(table, file=f)


name = 'label-frequency-training'
caption = 'Frequency of labels in the training set'
generate_image_frequency_table(df[df[cd.COL_TRAIN_VALIDATION] == cd.TRAINING], name, caption,
                               DIR_TABLES+name+'.tex')

name = 'label-frequency-validation'
caption = 'Frequency of labels in the validation set'
generate_image_frequency_table(df[df[cd.COL_TRAIN_VALIDATION] == cd.VALIDATION], name, caption,
                               DIR_TABLES+name+'.tex', pos_neg_only=True)

# Co-incidence of labels


def label_image_coincidence(df: pd.DataFrame) -> pd.DataFrame:
    labels = cd.OBSERVATION_OTHER + cd.OBSERVATION_PATHOLOGY
    stats = pd.DataFrame(index=labels, columns=labels)

    for label in labels:
        df_label = df[df[label] == 1]
        coincidences = [len(df_label[df_label[x] == 1]) for x in labels]
        stats.loc[label] = coincidences
    # Sanity check: 'No Finding' should not coincide with a pathology
    assert stats.loc[cd.OBSERVATION_NO_FINDING][cd.OBSERVATION_PATHOLOGY].sum() == 0
    return stats


name = 'label-coincidence'
caption = 'Coincidence of positive labels in the training set'
stats = label_image_coincidence(df[df[cd.COL_TRAIN_VALIDATION] == cd.TRAINING])
# Improve presentation
# Shorten long label names
stats.rename(index=SHORT_LABELS, inplace=True)
stats.rename(columns=SHORT_LABELS, inplace=True)
# Remove upper triangle (same as bottom triangle) to make it lighter
stats.values[np.triu_indices_from(stats, 0)] = ''
# Remove first row and last column (they are now empty)
stats.drop(labels=cd.OBSERVATION_NO_FINDING, axis='rows', inplace=True)
stats.drop(labels=cd.OBSERVATION_PATHOLOGY[-1], axis='columns', inplace=True)

table = stats.to_latex(column_format='r' * (stats.shape[1]+1),  # +1 for index
                       float_format=FLOAT_FORMAT, index_names=True,
                       caption=caption, label='tab:' + name, position='h!')

# Rotate column names
before = ' & ' + (' & ').join(stats.columns.tolist())
rotated = ' & ' + (' & ').join(['\\\\rotatebox{{90}}{{{}}}'.format(x)
                                for x in stats.columns.tolist()])
table = re.sub(' & {}.* & {}'.format(stats.columns[0], stats.columns[-1]), rotated, table, count=1)

# Horizontal separators to  make it easier to read
table = re.sub(r'^Consolidation',
               r'\\midrule[0.2pt]\nConsolidation', table, count=1, flags=re.MULTILINE)
table = re.sub(r'^Lung Opacity',
               r'\\midrule[0.2pt]\nLung Opacity', table, count=1, flags=re.MULTILINE)

# Use the page width
table = table.replace('\\begin{tabular}',
                      '\\begin{adjustbox}{width = 1\\textwidth}\n\\begin{tabular}')
table = table.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}')
table = table.replace('{table}', '{table*}')

file = DIR_TABLES+name+'.tex'
with open(file, 'w') as f:
    print(table, file=f)
