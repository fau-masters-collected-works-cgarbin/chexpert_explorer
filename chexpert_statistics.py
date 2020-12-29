"""CheXpert statistics."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chexpert_dataset as cd

sns.set_style("whitegrid")
sns.set_palette("Set2", 4, 0.75)

IMAGES = 'Images'


@st.cache
def get_dataset() -> pd.DataFrame:
    """Get a slimmed down version of the dataset.

    Implemented as a separate function to take advantage of Streamlit's caching.

    Returns:
        pd.DataFrame: The CheXPert dataframe.
    """
    chexpert = cd.CheXpert()
    chexpert.fix_dataset()
    # make it smaller to increase performance
    chexpert.df.drop('Path', axis='columns', inplace=True)
    df = chexpert.df

    # Hack for https://github.com/streamlit/streamlit/issues/47
    # Streamlit does not support categorical values
    # This undoes the preprocessing code, setting the columns back to string
    for c in [cd.COL_SEX, cd.COL_FRONTAL_LATERAL, cd.COL_AP_PA, cd.COL_AGE_GROUP,
              cd.COL_TRAIN_VALIDATION]:
        df[c] = df[c].astype('object')

    return df


st.set_page_config(page_title='CheXpert Statistics')

df = get_dataset()

st.write('Number of patients and images in the training and validation sets')
stats = df.groupby([cd.COL_TRAIN_VALIDATION], as_index=False, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
st.write(stats)

st.write('Same as above, split by sex')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_SEX], as_index=False,  observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
st.write(stats)

sns.catplot(y='Patients', x='Sex', col=cd.COL_TRAIN_VALIDATION, data=stats,
            kind='bar', sharey=False)
st.pyplot(plt)
sns.catplot(y='Images', x='Sex', col=cd.COL_TRAIN_VALIDATION, data=stats,
            kind='bar', sharey=False)
st.pyplot(plt)

st.write('Number of studies per patient')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID], as_index=False, observed=True).agg(
    Studies=(cd.COL_STUDY_NUMBER, pd.Series.nunique))
summary = stats.groupby([cd.COL_TRAIN_VALIDATION], as_index=False, observed=True).agg(
    Minimum=('Studies', 'min'),
    Maximum=('Studies', 'max'),
    Mean=('Studies', 'mean'),
    Median=('Studies', 'median'),
    Std=('Studies', 'std'))
st.write(summary)
st.write('Number of studies per quantile')
summary = stats[[cd.COL_TRAIN_VALIDATION, 'Studies']].groupby(
    [cd.COL_TRAIN_VALIDATION], as_index=True).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack().reset_index())

plt.clf()
ax = sns.countplot(x='Studies', data=stats, color='gray')
sns.despine(ax=ax)
ax.set(yscale="log")
plt.xticks(rotation=90)
plt.xlabel('Number of studies')
plt.ylabel('Number of patients (log)')
st.pyplot(plt)

st.write('Number of images per patient')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID], as_index=False, observed=True).agg(
    Images=(cd.COL_VIEW_NUMBER, 'count'))
summary = stats.groupby([cd.COL_TRAIN_VALIDATION], as_index=False).agg(
    Minimum=('Images', 'min'),
    Maximum=('Images', 'max'),
    Mean=('Images', 'mean'),
    Median=('Images', 'median'),
    Std=('Images', 'std'))
st.write(summary)
st.write('Number of images per quantile')
summary = stats[[cd.COL_TRAIN_VALIDATION, 'Images']].groupby(
    [cd.COL_TRAIN_VALIDATION], as_index=True, observed=True).quantile(
        [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack().reset_index())

plt.clf()
ax = sns.countplot(x='Images', data=stats, color='gray')
sns.despine(ax=ax)
ax.set(yscale="log")
plt.xticks(rotation=90)
plt.xlabel('Number of images')
plt.ylabel('Number of patients (log)')
st.pyplot(plt)
