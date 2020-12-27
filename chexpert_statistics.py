"""CheXpert statistics."""

import streamlit as st
import pandas as pd
import seaborn as sns
import chexpert_dataset as cd

sns.set_style("whitegrid")
sns.set_palette("Set3", 6, 0.75)

pd.options.display.float_format = '{:,.1f}'.format

IMAGES = 'Images'


@st.cache
def get_dataset() -> pd.DataFrame:
    """Get a slimmed down versio of the dataset.

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


df = get_dataset()

# ADD UNIT TESTS

# The number of images per patient/study
st.write('Images per patient/study')
df_psi = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID, cd.COL_STUDY_NUMBER],
                    as_index=False).agg(Images=(cd.COL_PATIENT_ID, 'count'))
st.dataframe(df_psi)


dfx = df.groupby([cd.COL_TRAIN_VALIDATION, cd.COL_PATIENT_ID, cd.COL_STUDY_NUMBER],
                 as_index=True).agg(Images=(cd.COL_PATIENT_ID, 'count'))
st.dataframe(dfx)


# Statistics for images/study
st.write('Images per study')
df_is_stats = df_psi.groupby(cd.COL_TRAIN_VALIDATION, as_index=False).agg(
    Minimum=(IMAGES, 'min'), Maximum=(IMAGES, 'max'), Std=(IMAGES, 'std'))
st.write(df_is_stats)

dfx2 = dfx.groupby(cd.COL_TRAIN_VALIDATION, as_index=True).agg(
    Minimum=(IMAGES, 'min'), Maximum=(IMAGES, 'max'), Std=(IMAGES, 'std'))
st.write(dfx2)


st.write('Images per train/validate set')
df_is_counts = df_psi.groupby([cd.COL_TRAIN_VALIDATION, IMAGES], as_index=False).agg(
    Count=(IMAGES, 'count'))
st.write(df_is_counts)

dfx3 = dfx.groupby([cd.COL_TRAIN_VALIDATION, IMAGES], as_index=True).agg(
    Count=(IMAGES, 'count'))
st.write(dfx3)
