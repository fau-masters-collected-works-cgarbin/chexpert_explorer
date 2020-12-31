"""CheXpert LaTex exporter.

Export CheXpert statistics and graphs to be imported in LaTex documents.

The goal is to automate the generation of all tables stastical tables used in papers, so that they
are accurate and can be regenerated quickly if the dataset is upgraded.
"""

import os
import re
from typing import List
import pandas as pd
import numpy as np
import chexpert_dataset as cd

# Destination directories, with path separator at the end to simplify the code
DIR_TABLES = os.path.join('..', 'chexpert-datasheet', 'tables') + os.sep
DIR_GRAPHS = os.path.join('..', 'chexpert-datasheet', 'graphs') + os.sep

IMAGES = 'Images'
PATIENTS = 'Patients'
FLOAT_FORMAT = '{:0,.1f}'.format
INT_FORMAT = '{:,}'.format

SHORT_OBSERVATION_NAMES = [('Enlarged Cardiomediastinum', 'Enlarged Card.')]

SEP_OBSERVATIONS = ['Consolidation', 'Lung Opacity']
SEP_TRAIN_VALIDATION = ['Validation']


def format_table(table: str, source_df: pd.DataFrame, file: str,
                 short_observation_name: bool = False, text_width: bool = False,
                 vertical_columns_names: bool = False, horizontal_separators: List[str] = None,
                 font_size: str = None):
    """Format a LaTeX table and saves it to a file.

    Args:
        table (str): The LaTeX table to be formatted.
        source_df (pd.DataFrame): The DataFrame used to generated the table.
        file (str): The base file name to save the table to. The directory and .tex extension are
            added in this function.
        short_observation_name (bool, optional): Shorten some of the observations names. Defaults
            to False.
        text_width (bool, optional): Use the full text width (for multi-column LaTeX templates).
            Defaults to False.
        vertical_columns_names (bool, optional): Rotate the columns names by 90 degrees. Defaults
            to False.
        horizontal_separators (List[str], optional): Add a horizontal separator before lines that
            start with these text.
        font_size (str, optional): Set the font size to the specified font, or use the default if
            ``None`` is specified. Defaults to None.
    """
    if text_width:
        table = table.replace('\\begin{tabular}',
                              '\\begin{adjustbox}{width = 1\\textwidth}\n\\begin{tabular}')
        table = table.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}')
        table = table.replace('{table}', '{table*}')

    if vertical_columns_names:
        # Assume columns names match the ones in the DataFrame
        rotated = ' & ' + (' & ').join(['\\\\rotatebox{{90}}{{{}}}'.format(x)
                                        for x in source_df.columns.tolist()])
        table = re.sub(' & {}.* & {}'.format(source_df.columns[0], source_df.columns[-1]),
                       rotated, table, count=1)

    for sep in horizontal_separators:
        table = re.sub(r'^{}'.format(sep), r'\\midrule[0.2pt]\n{}'.format(sep),
                       table, count=1, flags=re.MULTILINE)

    if font_size is not None:
        table = table.replace('\\centering', '\\{}\n\\centering'.format(font_size))

    if short_observation_name:
        # Not very memory efficient, but simple and sufficient for the text sizes we deal with
        for replacement in SHORT_OBSERVATION_NAMES:
            table = table.replace(*replacement)

    with open(DIR_TABLES + file + '.tex', 'w') as f:
        print(table, file=f)


chexpert = cd.CheXpert()
chexpert.fix_dataset()
# Make code a bit simpler
df = chexpert.df

# Count of patients and images in the training and validation datasets
NAME = 'patient-images-train-validate'
CAPTION = 'Number of patients and images'
stats = df.groupby([cd.COL_TRAIN_VALIDATION], as_index=True, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
assert stats.loc[cd.TRAINING][PATIENTS].sum() == cd.PATIENT_NUM_TRAINING
assert stats[PATIENTS].sum() == cd.PATIENT_NUM_TOTAL
assert stats[IMAGES].sum() == cd.IMAGE_NUM_TOTAL
stats.to_latex(buf=DIR_TABLES+NAME+'.tex',
               formatters=[INT_FORMAT] * stats.shape[1],
               float_format=FLOAT_FORMAT, index_names=False,
               caption=CAPTION, label='tab:'+NAME, position='h!')

# Summary statistic of images per patient
NAME = 'patient-images-stats-summary'
CAPTION = 'Summary statistics for images per patient'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID], as_index=True, observed=True).agg(
    Images=(cd.COL_VIEW_NUMBER, 'count'))
assert stats.loc[cd.TRAINING][IMAGES].sum() == cd.IMAGE_NUM_TRAINING
assert stats[IMAGES].sum() == cd.IMAGE_NUM_TOTAL
summary = stats.groupby([cd.COL_TRAIN_VALIDATION], as_index=True).agg(
    Min=(IMAGES, 'min'), Max=(IMAGES, 'max'), Median=(IMAGES, 'median'), Mean=(IMAGES, 'mean'),
    Std=(IMAGES, 'std'))
summary.to_latex(buf=DIR_TABLES+NAME+'.tex',
                 float_format=FLOAT_FORMAT, index_names=False,
                 caption=CAPTION, label='tab:'+NAME, position='h!')

# Binned number of images per patient (continuing from above, where the number of images was added)
NAME = 'patient-images-stats-distribution'
CAPTION = 'Distribution of images per patient'
bins = [0, 1, 2, 3, 10, 100]
bin_labels = ['1 image', '2 images', '3 images', '4 to 10 images', 'More than 10 images']
IMAGE_SUMMARY = 'Number of images'
stats[IMAGE_SUMMARY] = pd.cut(stats.Images, bins=bins, labels=bin_labels, right=True)

summary = stats.reset_index().groupby([IMAGE_SUMMARY], as_index=True, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique))
assert summary[PATIENTS].sum() == cd.PATIENT_NUM_TOTAL
summary.to_latex(buf=DIR_TABLES+NAME+'.tex',
                 formatters=[INT_FORMAT] * summary.shape[1],
                 float_format=FLOAT_FORMAT, index_names=True,
                 caption=CAPTION, label='tab:'+NAME, position='h!')

# Frequency of labels in the training and validation sets


def label_image_frequency(df: pd.DataFrame) -> pd.DataFrame:
    observations = cd.OBSERVATION_OTHER + cd.OBSERVATION_PATHOLOGY
    images_in_set = len(df[cd.COL_VIEW_NUMBER])
    ALL_LABELS = [cd.LABEL_POSITIVE, cd.LABEL_NEGATIVE, cd.LABEL_UNCERTAIN, cd.LABEL_NO_MENTION]
    COL_NAMES = ['Positive', '%', 'Negative', '%', 'Uncertain', '%', 'No mention', '%']
    stats = pd.DataFrame(index=observations, columns=COL_NAMES)
    for obs in observations:
        count = [len(df[df[obs] == x]) for x in ALL_LABELS]
        pct = [c*100/images_in_set for c in count]
        # Interleave count and percentage columns
        stats.loc[obs] = [x for t in zip(count, pct) for x in t]
    # Sanity check: check a few columns for the number of images
    cols_no_pct = [v for v in COL_NAMES if v != '%']
    assert stats.loc[cd.OBSERVATION_NO_FINDING][cols_no_pct].sum() == images_in_set
    assert stats.loc[cd.OBSERVATION_PATHOLOGY[1]][cols_no_pct].sum() == images_in_set
    return stats


def generate_image_frequency_table(df: pd.DataFrame, name: str, caption: str,
                                   pos_neg_only: bool = False) -> str:
    stats = label_image_frequency(df)
    if pos_neg_only:
        # Assume pos/neg count and % are the first columns
        stats = stats.iloc[:, :4]
    font_size = 'small' if pos_neg_only else 'scriptsize'

    table = stats.to_latex(column_format='l' + 'r' * stats.shape[1],
                           formatters=[INT_FORMAT, '{:.1%}'.format] * (stats.shape[1]//2),
                           float_format=FLOAT_FORMAT, index_names=True,
                           caption=caption, label='tab:'+name, position='h!')
    format_table(table, stats, name, short_observation_name=True, text_width=not pos_neg_only,
                 horizontal_separators=SEP_OBSERVATIONS, font_size=font_size)


NAME = 'label-frequency-training'
CAPTION = 'Frequency of labels in the training set'
generate_image_frequency_table(df[df[cd.COL_TRAIN_VALIDATION] == cd.TRAINING], NAME, CAPTION)

NAME = 'label-frequency-validation'
CAPTION = 'Frequency of labels in the validation set'
generate_image_frequency_table(df[df[cd.COL_TRAIN_VALIDATION] == cd.VALIDATION], NAME, CAPTION,
                               pos_neg_only=True)

# Co-incidence of labels


def label_image_coincidence(df: pd.DataFrame) -> pd.DataFrame:
    labels = cd.OBSERVATION_OTHER + cd.OBSERVATION_PATHOLOGY
    stats = pd.DataFrame(index=labels, columns=labels)

    for label in labels:
        df_label = df[df[label] == 1]
        coincidences = [len(df_label[df_label[other_label] == 1]) for other_label in labels]
        stats.loc[label] = coincidences
    # Sanity check: 'No Finding' should not coincide with a pathology
    assert stats.loc[cd.OBSERVATION_NO_FINDING][cd.OBSERVATION_PATHOLOGY].sum() == 0
    return stats


NAME = 'label-coincidence'
CAPTION = 'Coincidence of positive labels in the training set'
stats = label_image_coincidence(df[df[cd.COL_TRAIN_VALIDATION] == cd.TRAINING])
# Remove upper triangle (same as bottom triangle) to make it easier to follow
stats.values[np.triu_indices_from(stats, 0)] = ''
# Remove first row and last column (they are now empty)
stats.drop(labels=cd.OBSERVATION_NO_FINDING, axis='rows', inplace=True)
stats.drop(labels=cd.OBSERVATION_PATHOLOGY[-1], axis='columns', inplace=True)

table = stats.to_latex(column_format='r' * (stats.shape[1]+1),  # +1 for index
                       float_format=FLOAT_FORMAT, index_names=True,
                       caption=CAPTION, label='tab:'+NAME, position='h!')
format_table(table, stats, NAME, text_width=True, short_observation_name=True,
             vertical_columns_names=True, horizontal_separators=SEP_OBSERVATIONS)


NAME = 'demographic-by-set-sex'
CAPTION = 'Images and patients by sex'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_SEX], as_index=True,  observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
table = stats.to_latex(formatters=[INT_FORMAT] * stats.shape[1],
                       float_format=FLOAT_FORMAT, index_names=True,
                       caption=CAPTION, label='tab:'+NAME, position='h!')
format_table(table, stats, NAME, horizontal_separators=SEP_TRAIN_VALIDATION)

NAME = 'demographic-by-set-age-group'
CAPTION = 'Images and patients by age group'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_AGE_GROUP], as_index=True,  observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
table = stats.to_latex(formatters=[INT_FORMAT] * stats.shape[1],
                       float_format=FLOAT_FORMAT, index_names=True,
                       caption=CAPTION, label='tab:'+NAME, position='h!')
format_table(table, stats, NAME, horizontal_separators=SEP_TRAIN_VALIDATION, font_size='small')

NAME = 'demographic-by-set-sex-age-group'
CAPTION = 'Images and patients by sex and age group'
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_AGE_GROUP, cd.COL_SEX], as_index=True,
                   observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
stats = stats.unstack(fill_value=0).reorder_levels([1, 0], axis='columns')
table = stats.to_latex(formatters=[INT_FORMAT] * stats.shape[1],
                       float_format=FLOAT_FORMAT, index_names=True,
                       caption=CAPTION, label='tab:'+NAME, position='h!')
format_table(table, stats, NAME, horizontal_separators=SEP_TRAIN_VALIDATION, font_size='small',
             text_width=True)
