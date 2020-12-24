"""Explore the CheXpert dataset with Streamlit (https://www.streamlit.io/).

The code assumes that the dataset has been uncompressed into the same directory this file is in.

Run with: streamlit run chexpert-explorer.py
"""

from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess as p


sns.set_style("whitegrid")
sns.set_palette("Set3", 6, 0.75)

ALL_LABELS = '(All)'


@st.cache
def get_percentages(df: pd.DataFrame, totals: bool, percentages: str) -> pd.DataFrame:
    """Return a percentage version of the DataFrame, across rows or columns, as specified.

    Based on https://stackoverflow.com/a/42006745.

    Args:
        df (pd.DataFrame): The DataFrame with the raw numbers.
        totals (bool): Whether the DataFrame has a "totals" row and column (when aggregated with
            "margins" set to True).
        percentages (str): Whether to calculate percentages by ``row`` or ``column``.

    Returns:
        pd.DataFrame: The DataFrame in percentages.
    """
    # Account for a "Totals" row if one was requested
    divider = 2 if totals else 1
    sum_axis = 'rows' if percentages == 'columns' else 'columns'
    df_pct = df.copy()
    cols = df_pct.columns
    df_pct[cols] = df_pct[cols].div(df_pct[cols].sum(
        axis=sum_axis)/divider, axis=percentages).multiply(100)
    return df_pct


@st.cache
def get_pivot_table(labels: List[str], rows: List[str], columns: List[str],
                    totals: bool = False) -> pd.DataFrame:
    """Get a pivot table with the selected labels.

    All operations on the dataset are done here to take advantage of Streamlit's cache. If we
    modify the dataset after returning it, we will get warnings that are mutating a cached object.

    Args:
        labels (List[str]): The list of labels to select from the dataset, or an empty list to
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
    for label in labels:
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
    labels.insert(0, ALL_LABELS)
    return labels


@st.cache
def get_flattened_df(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Get a flattened version of the DataFrame.

    The flattened version of the DataFrames has only one column with values (the count of images).
    The other columns in the DataFrame are descriptions of that value. All indices, including
    multi-indices, are stored as columns.

    Args:
        df_agg (pd.DataFrame): The aggregrated DataFrame, created with pivot_table or groupby.

    Returns:
        pd.DataFrame: The flattened DataFrame, with indices moved to columns.
    """
    # Flatten the dataframe
    df = df_agg.stack(list(range(df_agg.columns.nlevels)))
    df = df.reset_index()
    # Rename the count column to something more meaningful
    df.rename(columns={0: 'Count of images'}, inplace=True)
    return df


def show_graph(df_agg: pd.DataFrame):
    """Show a graph for the aggregated DataFrame.

    The graph is a collection of bar plots, categorized by the columns, left to right.  The second
    column is always used as the hue. The other columns are used for rows and columns in the graph.

    Args:
        df_agg (pd.DataFrame): The aggregrated DataFrame to graph.
    """
    df = get_flattened_df(df_agg)

    columns = df.columns
    num_columns = len(columns)
    if num_columns == 3:
        sns.barplot(x=columns[0], y=columns[2], hue=columns[1], data=df)
        st.pyplot(plt)
    elif num_columns == 4:
        # Let the user change the number of graphs per row
        max_col_wrap = len(df[columns[2]].unique())
        col_wrap = st.number_input('Graphs per row', min_value=1, max_value=max_col_wrap,
                                   value=max_col_wrap)
        # When showing one graph per row, let the y axis adjust to the data to show more details
        sharey = col_wrap > 1
        if not sharey:
            st.write('Y axis not to the same scale')
        sns.catplot(x=columns[0], y=columns[3], hue=columns[1], col=columns[2], data=df, kind='bar',
                    col_wrap=col_wrap, sharey=sharey)
        st.pyplot(plt)
    elif num_columns == 5:
        sns.catplot(x=columns[0], y=columns[4], hue=columns[1], col=columns[2], row=columns[3],
                    data=df, kind='bar')
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

percentages = st.sidebar.radio('Add percentages across', ('Rows', 'Columns', 'No percentages'))

if not rows and not columns:
    st.write('Select rows and columns')
elif not set(rows).isdisjoint(columns):
    st.write('Rows and columns cannot have the same values')
else:
    # Remove the special "all labels" label if it's present in the list
    adjusted_labels = labels[:]  # make a copy
    if ALL_LABELS in adjusted_labels:
        adjusted_labels.remove(ALL_LABELS)

    totals = False
    df_agg = get_pivot_table(adjusted_labels, rows, columns, totals=totals)
    if df_agg.empty:
        st.write('There are no images with this combination of filters.')
    else:
        pct_lower = percentages.lower()
        if pct_lower in ['rows', 'columns']:
            pvt_pct = get_percentages(df_agg, totals, pct_lower)
            # Combine the percentanges with the value
            df_agg = df_agg.combine(pvt_pct,
                                    lambda s1, s2: ['{:,d} ({:5.1f})'.format(
                                        int(v1), v2) for v1, v2 in zip(s1, s2)])

        st.write(df_agg)
        show_graph(get_pivot_table(adjusted_labels, rows, columns, totals=False))

    # pct = get_percentages(df_agg, totals=False, percentages=)
    # st.write('Unstacked')
    # st.write(df_agg.unstack())
    # st.write(get_percentages)
