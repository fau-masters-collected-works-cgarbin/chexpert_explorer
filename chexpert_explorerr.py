"""CheXpert statistics.

Different views into CheXpert to understand its composition.
"""
from typing import List
import streamlit as st
import pandas as pd
import chexpert_dataset as cxd
import chexpert_statistics as cxs

ALL_OPTIONS = '(All)'

st.set_page_config(page_title='CheXpert Statistics')
st.markdown('# CheXpert Statistics')


@st.cache
def _get_dataset() -> pd.DataFrame:
    """Wrap the dataset around Straeamlit's cache to speed up the code a bit."""
    cxdata = cxd.CheXpertDataset()
    cxdata.fix_dataset()
    _df = cxdata.df

    # Hack for https://github.com/streamlit/streamlit/issues/47
    # Streamlit does not support categorical values
    # This undoes the preprocessing code, setting the columns back to string
    for c in [cxd.COL_SEX, cxd.COL_FRONTAL_LATERAL, cxd.COL_AP_PA, cxd.COL_AGE_GROUP,
              cxd.COL_TRAIN_VALIDATION]:
        _df[c] = _df[c].astype('object')

    # make it smaller to increase performance
    _df.drop('Path', axis='columns', inplace=True)

    return _df


@st.cache
def _get_observations() -> List[str]:
    """Get a list of observations to show to the user, extracted from the dataset column names.

    Observations are returned with other findings at the other, followed by pathologies in
    alphabetical order. An explicit option to select all observations is added as the first entry
    (even though selecting none means "show all", this option makes it clear to the user).

    Returns:
        List[str]: List of observations to show to the user.
    """
    observations = cxd.OBSERVATION_OTHER + cxd.OBSERVATION_PATHOLOGY
    observations.insert(0, ALL_OPTIONS)
    return observations


df_complete = _get_dataset()

observations = st.multiselect(
    'Show statistics for images with positive label for these observations (select one or more)',
    _get_observations(), default=ALL_OPTIONS)
# Warn the user that "all observations" is ignored when used with other observations
if ALL_OPTIONS in observations and len(observations) > 1:
    st.write('Ignoring "{}" when used with other observations'.format(ALL_OPTIONS))

# Remove the special "All" option from list of observations
adjusted_observations = observations[:]  # make a copy to not affect the UI
if ALL_OPTIONS in adjusted_observations:
    adjusted_observations.remove(ALL_OPTIONS)

# Filter the dataset based on selected observations
df = df_complete
for obs in adjusted_observations:
    df = df[df[obs] == cxd.LABEL_POSITIVE]

if df.shape[0] == 0:
    st.write('There are no images with this combination of observations')
else:
    filtered = df.shape[0] != df_complete.shape[0]

    st.markdown('## Number of patients, studies, and images')
    stats = cxs.patient_study_image_count(df, filtered=filtered)
    # Long format, without the "Counts" column index
    stats = stats.unstack().reorder_levels([1, 0], axis='columns').droplevel(1, axis='columns')
    st.write(stats)

    st.markdown('## Summary statistics for studies per patient')
    summary = cxs.studies_summary_stats(df)
    st.write(summary)

    st.markdown('### Number of studies per quantile')
    stats = cxs.studies_per_patient(df).reset_index()
    summary = stats[[cxd.COL_TRAIN_VALIDATION, cxs.STUDIES]].groupby(
        [cxd.COL_TRAIN_VALIDATION], as_index=True).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    st.write(summary.unstack())

    st.markdown('## Summary statistics for images per patient')
    summary = cxs.images_summary_stats(df)
    st.write(summary)

    st.markdown('### Number of images per quantile')
    stats = cxs.images_per_patient(df).reset_index()
    summary = stats[[cxd.COL_TRAIN_VALIDATION, cxs.IMAGES]].groupby(
        [cxd.COL_TRAIN_VALIDATION], as_index=True, observed=True).quantile(
            [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    st.write(summary.unstack().reset_index())

    st.markdown('### Binned number of images')
    summary = cxs.images_per_patient_binned(df, filtered=filtered)
    # Hack for https://github.com/streamlit/streamlit/issues/47
    summary = summary.reset_index()
    for c in [cxd.COL_TRAIN_VALIDATION, 'Number of images']:
        summary[c] = summary[c].astype('object')
    st.write(summary)

    st.markdown('## Demographics')

    st.markdown('### Number of patients and images by sex')
    stats = cxs.images_per_patient_sex(df)
    st.write(stats)

    st.markdown('### Number of patients and images by age group')
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP],
                       as_index=True, observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    st.write(stats)

    st.markdown('### Number of patients and images by sex and age group')
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP, cxd.COL_SEX], as_index=True,
                       observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    stats = stats.unstack(fill_value=0).reorder_levels([1, 0], axis='columns')
    st.write(stats)
