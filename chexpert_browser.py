"""CheXpert statistics.

Different views into CheXpert to understand its composition.
"""

import streamlit as st
import pandas as pd
import chexpert_dataset as cxd
import chexpert_statistics as cxs

cxdata = cxd.CheXpertDataset()
cxdata.fix_dataset()
df = cxdata.df  # Shortcut for smaller code

# Hack for https://github.com/streamlit/streamlit/issues/47
# Streamlit does not support categorical values
# This undoes the preprocessing code, setting the columns back to string
for c in [cxd.COL_SEX, cxd.COL_FRONTAL_LATERAL, cxd.COL_AP_PA, cxd.COL_AGE_GROUP,
          cxd.COL_TRAIN_VALIDATION]:
    df[c] = df[c].astype('object')

st.set_page_config(page_title='CheXpert Statistics')
st.markdown('# CheXpert Statistics')

st.markdown('## Number of patients, studies, and images')
stats = cxs.patient_study_image_count(df)
# Long format, without the "Counts" column index
stats = stats.unstack().reorder_levels([1, 0], axis='columns').droplevel(1, axis='columns')
st.write(stats)

st.markdown('### Binned number of images')
summary = cxs.images_per_patient_binned(df)
# Hack for https://github.com/streamlit/streamlit/issues/47
summary = summary.reset_index()
for c in [cxd.COL_TRAIN_VALIDATION, cxs.IMAGES]:
    summary[c] = summary[c].astype('object')
st.write(summary)

st.markdown('## Summary statistics for studies per patient')
summary = cxs.studies_summary_stats(df)
st.write(summary)

st.markdown('### Number of studies per quantile')
stats = cxs.studies_per_patient(df).reset_index()
summary = stats[[cxd.COL_TRAIN_VALIDATION, 'Studies']].groupby(
    [cxd.COL_TRAIN_VALIDATION], as_index=True).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack())

st.markdown('## Summary statistics for images per patient')
summary = cxs.images_summary_stats(df)
st.write(summary)

st.markdown('### Number of images per quantile')
stats = cxs.images_per_patient(df).reset_index()
summary = stats[[cxd.COL_TRAIN_VALIDATION, 'Images']].groupby(
    [cxd.COL_TRAIN_VALIDATION], as_index=True, observed=True).quantile(
        [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack().reset_index())


st.markdown('## Demographics')

st.markdown('### Number of patients and images by sex')
stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_SEX], as_index=False,  observed=True).agg(
    Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cxd.COL_VIEW_NUMBER, 'count'))
st.write(stats)

st.markdown('### Number of patients and images by age group')
stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP], as_index=True, observed=True).agg(
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
