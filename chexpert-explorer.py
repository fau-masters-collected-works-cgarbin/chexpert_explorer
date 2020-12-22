"""Explore the CheXpert dataset with Streamlit (https://www.streamlit.io/).

The code assumes that the dataset has been uncompressed into the same directory this file is in.

Run with: streamlit run chexpert-explorer.py
"""

from typing import List
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess as p


sns.set_style("whitegrid")
sns.set_palette("Set3", 6, 0.75)

ALL_LABELS = '(All)'


@st.cache
def get_pivot_table(labels: List[str], rows: List[str], columns: List[str],
                    totals: bool = False) -> pd.DataFrame:
    """Get a pivot table with the selected labels.

    All operations on the dataset are done here to take advantage of Streamlit's cache. If we
    modify the dataset after returning it here, we will get warnings that are mutating a cached
    object.

    Args:
        labels (List[str]): The list labels to select from the dataset, or an empty list to
                            select all labels.
        rows (List[str]): The list of dataset fields to use as the rows (indices).
        columns (List[str]): The list of dataset fiels to use as columns.
        totals (bool): Set to True to get totals by row and column

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
    for c in [p.COL_SEX, p.COL_FRONTAL_LATERAL, p.COL_AP_PA, p.COL_AGE_GROUP,
              p.COL_TRAIN_VALIDATION]:
        df[c] = df[c].astype('object')

    # Keep the rows that have the selected labels
    for label in adjusted_labels:
        df = df[df[label] == 1]

    # Add a column to aggregate on
    df['count'] = 1

    # Catch the case where the combination of filters results in no images
    if df.empty:
        return df

    pvt = pd.pivot_table(df, values='count', index=rows, columns=columns, aggfunc=sum, fill_value=0,
                         margins=totals, margins_name='Total')
    return pvt


@st.cache
def get_labels() -> List[str]:
    """Get a list of labels to show to the user, extracted from the dataset column names.

    Labels are returned in alphabetical order. An explicit option to select all labels is added as
    the first entry (even though selecting no labels means "show all", this option makes it clear to
    the user).

    Returns:
        List[str]: List of labels to show to the user.
    """
    labels = sorted(p.COL_LABELS)
    # Insert an explicit choice for all labels - even though selecting no labels means "show all",
    # this option makes it clear to the user
    labels.insert(0, ALL_LABELS)
    return labels


def show_graph(df_agg: pd.DataFrame):
    # Flatten the dataframe
    df = df_agg.stack(list(range(df_agg.columns.nlevels)))
    df = df.reset_index()
    # Rename the count column to something more meaningful
    df.rename(columns={0: 'Count of images'}, inplace=True)

    columns = df.columns
    num_columns = len(columns)
    if num_columns == 3:
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.catplot(x=columns[0], y=columns[-1], hue=columns[1], kind="bar", data=df, ax=ax)
        st.pyplot(plt)


st.set_page_config(page_title='CheXpert Explorer')

ROW_COLUMNS = [p.COL_SEX, p.COL_AGE_GROUP, p.COL_TRAIN_VALIDATION, p.COL_FRONTAL_LATERAL]
rows = st.sidebar.multiselect('Select rows', ROW_COLUMNS)
columns = st.sidebar.multiselect('Select columns', ROW_COLUMNS)

labels = st.sidebar.multiselect('Show count of images with these labels (select one or more)',
                                get_labels(), default=ALL_LABELS)
# Warn the user that "all labels" is ignored when used with other labels
if ALL_LABELS in labels and len(labels) > 1:
    st.sidebar.write('Ignoring "{}" when used with other labels'.format(ALL_LABELS))

if not rows and not columns:
    st.write('Select rows and columns')
elif not set(rows).isdisjoint(columns):
    st.write('Rows and columns cannot have the same values')
else:
    # Remove the special "all labels" label if it's present in the list
    adjusted_labels = labels[:]  # make a copy
    if ALL_LABELS in adjusted_labels:
        adjusted_labels.remove(ALL_LABELS)

    df_agg = get_pivot_table(adjusted_labels, rows, columns, totals=True)
    if df_agg.empty:
        st.write('There are no images with this combination of filters.')
    else:
        st.write(df_agg)
        show_graph(get_pivot_table(adjusted_labels, rows, columns, totals=False))

        # Code for experiments

        def show_dataframe_details(df, title):
            st.write(title)
            st.write(df)
            st.write('Indices')
            st.write(df.index)
            st.write('Columns')
            st.write(df.columns)

        agg = st.checkbox('Aggreated dataframe')
        if agg:
            show_dataframe_details(df_agg, 'Aggregated dataframe')
        ri = st.checkbox('Reset Index')
        if ri:
            show_dataframe_details(df_agg.reset_index(), 'Reset Index')
        df_agg_u = pd.DataFrame(df_agg.unstack())
        unstacked = st.checkbox('Unstacked')
        if unstacked:
            show_dataframe_details(df_agg_u, 'Unstacked')
        uri = st.checkbox('Unstacked - Reset Index')
        if uri:
            show_dataframe_details(df_agg_u.reset_index(), 'Unstacked - Reset Index')
        stack_all_ri = st.checkbox('Stack all levels, reset index')
        if stack_all_ri:
            df = df_agg.stack(list(range(df_agg.columns.nlevels)))
            df = df.reset_index()
            show_dataframe_details(df, 'Stack all levels, reset index')
