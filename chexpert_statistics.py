"""CheXpert statistics.

Different views into CheXpert to understand its composition.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chexpert_dataset as cd

sns.set_style("whitegrid")
sns.set_palette("Set2", 4, 0.75)

IMAGES = 'Images'

chexpert = cd.CheXpert()
chexpert.fix_dataset()
df = chexpert.df  # Shortcut for smaller code

# Hack for https://github.com/streamlit/streamlit/issues/47
# Streamlit does not support categorical values
# This undoes the preprocessing code, setting the columns back to string
for c in [cd.COL_SEX, cd.COL_FRONTAL_LATERAL, cd.COL_AP_PA, cd.COL_AGE_GROUP,
          cd.COL_TRAIN_VALIDATION]:
    df[c] = df[c].astype('object')


st.set_page_config(page_title='CheXpert Statistics')
st.markdown('# CheXpert Statistics')

st.markdown('## Number of patients, studies, and images')
stats = chexpert.patient_study_image_count()
# Long format, without the "Counts" column index
stats = stats.unstack().reorder_levels([1, 0], axis='columns').droplevel(1, axis='columns')
st.write(stats)

# Note about reset_index(): this makes the aggregations columns, not indices so that 1) Streamlit
# display columns names (it doesn't display index names), and 2) we can use as x/y in plots (I
# didn't find a way to use indices as x/y directly)

st.markdown('## Summary statistics for studies per patient')
summary = chexpert.studies_summary_stats()
st.write(summary)

st.markdown('### Number of studies per quantile')
stats = chexpert.studies_per_patient().reset_index()
summary = stats[[cd.COL_TRAIN_VALIDATION, 'Studies']].groupby(
    [cd.COL_TRAIN_VALIDATION], as_index=True).quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack().reset_index())

# plt.clf()
# ax = sns.countplot(x='Studies', data=stats, color='gray')
# sns.despine(ax=ax)
# ax.set(yscale="log")
# plt.xticks(rotation=90)
# plt.xlabel('Number of studies')
# plt.ylabel('Number of patients (log)')
# st.pyplot(plt)

st.markdown('## Summary statistics for images per patient')
summary = chexpert.images_summary_stats()
st.write(summary)

st.markdown('### Number of images per quantile')
stats = chexpert.images_per_patient().reset_index()
summary = stats[[cd.COL_TRAIN_VALIDATION, 'Images']].groupby(
    [cd.COL_TRAIN_VALIDATION], as_index=True, observed=True).quantile(
        [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
st.write(summary.unstack().reset_index())

st.markdown('### Binned number of images')
bins = [0, 1, 2, 3, 10, 100]
bin_labels = ['1 image', '2 images', '3 images', '4 to 10 images', 'More than 10 images']
IMAGE_SUMMARY = 'Number of images'
stats[IMAGE_SUMMARY] = pd.cut(stats.Images, bins=bins,
                              labels=bin_labels, right=True).astype('object')
summary = stats.reset_index().groupby([IMAGE_SUMMARY], as_index=True, observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique))
st.write(summary)

# plt.clf()
# ax = sns.countplot(x='Images', data=stats, color='gray')
# sns.despine(ax=ax)
# ax.set(yscale="log")
# plt.xticks(rotation=90)
# plt.xlabel('Number of images')
# plt.ylabel('Number of patients (log)')
# st.pyplot(plt)

st.markdown('## Demographics')

st.markdown('### Number of patients and images by sex')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_SEX], as_index=False,  observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
st.write(stats)

st.markdown('### Number of patients and images by age group')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_AGE_GROUP], as_index=True,  observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
st.write(stats)

st.markdown('### Number of patients and images by sex and age group')
stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_AGE_GROUP, cd.COL_SEX], as_index=True,
                   observed=True).agg(
    Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
    Images=(cd.COL_VIEW_NUMBER, 'count'))
stats = stats.unstack(fill_value=0).reorder_levels([1, 0], axis='columns')
st.write(stats)

# Redo stats without index to use columns names in the plot
# stats = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_AGE_GROUP, cd.COL_SEX], as_index=False,
#                    observed=True).agg(
#     Patients=(cd.COL_PATIENT_ID, pd.Series.nunique),
#     Images=(cd.COL_VIEW_NUMBER, 'count'))
# plt.clf()
# sns.catplot(y='Patients', x='Sex', hue=cd.COL_AGE_GROUP, col=cd.COL_TRAIN_VALIDATION, data=stats,
#             kind='bar', sharey=False)
# st.pyplot(plt)
# sns.catplot(y='Images', x='Sex', hue=cd.COL_AGE_GROUP, col=cd.COL_TRAIN_VALIDATION, data=stats,
#             kind='bar', sharey=False)
# st.pyplot(plt)
