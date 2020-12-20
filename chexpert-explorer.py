"""Explore the CheXpert dataset with Streamlit (https://www.streamlit.io/).

The code assumes that the dataset has been uncompressed into the same directory this file is in.

Run with: streamlit run chexpert-explorer.py
"""

import pandas as pd
import streamlit as st
import seaborn as sns
import preprocess as p


sns.set_style("whitegrid")


@st.cache
def get_pivot_table(labels):
    """Get a pivot table with the selected labels.

    All operations on the dataset are done here to take advantage of Streamlit's cache. If we
    modify the dataset after returning it here, we will get warnings that are mutating a cached
    object.

    Args:
        labels (List[str]): The list labels to select from the dataset, or an empty list to
                            select all labels.

    Returns:
        [pd.DataFrame]: A pivot table with the number of images for the selected labels.
    """
    # Start with the fulll dataset
    df = p.get_augmented_dataset()
    p.fix_dataset(df)
    df.drop('Path', axis='columns', inplace=True)  # make it smaller to increase performance

    # Hack for https://github.com/streamlit/streamlit/issues/47
    # Streamlit does not support categorical values
    # This undoes the preprocessing code, setting the columns back to string
    for c in ['Sex', 'Frontal/Lateral', 'AP/PA', p.COL_AGE_GROUP, p.COL_TRAIN_VALIDATION]:
        df[c] = df[c].astype('object')

    # Keep the rows that have the selected labels
    for label in labels:
        df = df[df[label] == 1]

    # Add a column to aggregate on
    df['count'] = 1

    pvt = pd.pivot_table(df, values='count', index=[p.COL_AGE_GROUP],
                         columns=[p.COL_TRAIN_VALIDATION, 'Sex'], aggfunc=sum, fill_value=0)
    return pvt


labels = st.multiselect('Show count of images with these labels',
                        p.COL_LABELS, default='No Finding')

df_agg = get_pivot_table(labels)
st.write(df_agg)
