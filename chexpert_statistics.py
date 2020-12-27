"""CheXpert statistics."""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import preprocess as p

sns.set_style("whitegrid")
sns.set_palette("Set3", 6, 0.75)

pd.options.display.float_format = '{:,.1f}'.format

IMAGES = 'Images'


@st.cache
def get_dataset():
    df = p.get_augmented_dataset()
    p.fix_dataset(df)

    df.drop('Path', axis='columns', inplace=True)  # make it smaller to increase performance

    # Hack for https://github.com/streamlit/streamlit/issues/47
    # Streamlit does not support categorical values
    # This undoes the preprocessing code, setting the columns back to string
    for c in [p.COL_SEX, p.COL_FRONTAL_LATERAL, p.COL_AP_PA, p.COL_AGE_GROUP,
              p.COL_TRAIN_VALIDATION]:
        df[c] = df[c].astype('object')

    return df


df = get_dataset()

# ADD UNIT TESTS

# The number of images per patient/study
st.write('Images per patient/study')
df_psi = df.groupby([p.COL_TRAIN_VALIDATION, p.COL_PATIENT_ID, p.COL_STUDY_NUMBER], as_index=False).agg(
    Images=(p.COL_PATIENT_ID, 'count'))
st.dataframe(df_psi)


dfx = df.groupby([p.COL_TRAIN_VALIDATION, p.COL_PATIENT_ID, p.COL_STUDY_NUMBER], as_index=True).agg(
    Images=(p.COL_PATIENT_ID, 'count'))
st.dataframe(dfx)


# Statistics for images/study
st.write('Images per study')
df_is_stats = df_psi.groupby(p.COL_TRAIN_VALIDATION, as_index=False).agg(
    Minimum=(IMAGES, 'min'), Maximum=(IMAGES, 'max'), Std=(IMAGES, 'std'))
st.write(df_is_stats)

dfx2 = dfx.groupby(p.COL_TRAIN_VALIDATION, as_index=True).agg(
    Minimum=(IMAGES, 'min'), Maximum=(IMAGES, 'max'), Std=(IMAGES, 'std'))
st.write(dfx2)


st.write('Images per train/validate set')
df_is_counts = df_psi.groupby([p.COL_TRAIN_VALIDATION, IMAGES], as_index=False).agg(
    Count=(IMAGES, 'count'))
st.write(df_is_counts)

dfx3 = dfx.groupby([p.COL_TRAIN_VALIDATION, IMAGES], as_index=True).agg(
    Count=(IMAGES, 'count'))
st.write(dfx3)
