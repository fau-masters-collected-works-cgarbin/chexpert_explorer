"""CheXpert LaTex exporter.

Export CheXpert statistics and graphs to be imported in LaTex documents.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chexpert_dataset as cd

sns.set_style("whitegrid")
sns.set_palette("Set2", 4, 0.75)

# Destination directories, with path separator at the end to simplify the code
DIR_TABLES = os.path.join('..', 'chexpert-datasheet', 'tables') + os.sep
DIR_GRAPHS = os.path.join('..', 'chexpert-datasheet', 'graphs') + os.sep

IMAGES = 'Images'
FLOAT_FORMAT = '{:0,.1f}'.format
INT_FORMAT = '{:,}'.format

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
stats.to_latex(buf=DIR_TABLES+name+'.tex',
               formatters=[INT_FORMAT] * stats.shape[1],
               float_format=FLOAT_FORMAT, index_names=False,
               caption=caption, label='tab:' + name, position='h!')

# Summary statistic of images per patient
name = 'patient-images-stats-summary'
caption = 'Summary statistics for images per patient'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID], as_index=True, observed=True).agg(
    Images=(cd.COL_VIEW_NUMBER, 'count'))
summary = stats.groupby([cd.COL_TRAIN_VALIDATION], as_index=True).agg(
    Min=('Images', 'min'),
    Max=('Images', 'max'),
    Mean=('Images', 'mean'),
    Median=('Images', 'median'),
    Std=('Images', 'std'))
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
print(summary.columns)
summary.to_latex(buf=DIR_TABLES+name+'.tex',
                 formatters=[INT_FORMAT] * summary.shape[1],
                 float_format=FLOAT_FORMAT, index_names=True,
                 caption=caption, label='tab:' + name, position='h!')
