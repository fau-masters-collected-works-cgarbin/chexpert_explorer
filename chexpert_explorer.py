"""Explore the CheXpert dataset with Streamlit (https://www.streamlit.io/).

The code assumes that the dataset has been uncompressed into the same directory this file is in.

Run with: streamlit run chexpert-explorer.py
"""

from typing import List
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import chexpert_dataset as cd
import dfutils as du


sns.set_style("whitegrid")
sns.set_palette("Set3", 6, 0.75)

ALL_LABELS = '(All)'
TOTAL = 'Total'


@st.cache
def get_images_count(labels: List[str], rows: List[str], columns: List[str],
                     totals: bool = False, percentages: str = None,
                     format: bool = False) -> pd.DataFrame:
    """Get a pivot table with the image count for the selected filters.

    All operations on the dataset are done here to take advantage of Streamlit's cache. If we
    modify the dataset after returning it, we will get warnings that are mutating a cached object
    (not to mention potential bugs that are obscure to debug).

    Args:
        labels (List[str]): The list of labels to select from the dataset, or an empty list to
            select all labels.
        rows (List[str]): The list of dataset fields to use as the rows (indices).
        columns (List[str]): The list of dataset fiels to use as columns.
        totals (bool): Set to True to get totals by row and column
        percentages: add percentages by ``rows`` or ``columns``, or None to not add percentages.
        format: whether to format the numbers.

    Returns:
        [pd.DataFrame]: A pivot table with the number of images for the selected labels.
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

    # Keep the rows that have the selected labels
    for label in labels:
        df = df[df[label] == 1]

    # Add a column to aggregate on
    df['count'] = 1

    # Catch the case where the combination of filters results in no images
    if df.empty:
        return df

    pvt = pd.pivot_table(df, values='count', index=rows, columns=columns, aggfunc=sum, fill_value=0,
                         margins=totals, margins_name=TOTAL, dropna=False)
    pvt.fillna(0, inplace=True)

    if percentages in ['rows', 'columns']:
        pvt = du.merge_percentages(pvt, percentages, TOTAL)

    # Streamlit does not yet support formatting for multiindex dataframes
    # It will be added with https://github.com/streamlit/streamlit/issues/956
    # Until that issue is resolved, we need to format the numbers ourselves
    if format:
        pvt = pvt.applymap(lambda x: '{:,.0f}'.format(x))

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
    labels = sorted(cd.COL_LABELS)
    labels.insert(0, ALL_LABELS)
    return labels


def show_graph(df_agg: pd.DataFrame):
    """Show a graph for the aggregated DataFrame.

    The graph is a collection of bar plots, categorized by the columns, left to right.  The second
    column is always used as the hue. The other columns are used for rows and columns in the graph.

    Args:
        df_agg (pd.DataFrame): The aggregrated DataFrame to graph.
    """
    df = du.get_flattened_df(df_agg, reset_index=True)

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
        sharey = st.checkbox('Share Y axes')
        sns.catplot(x=columns[0], y=columns[3], hue=columns[1], col=columns[2], data=df, kind='bar',
                    col_wrap=col_wrap, sharey=sharey)
        st.pyplot(plt)
    elif num_columns == 5:
        sns.catplot(x=columns[0], y=columns[4], hue=columns[1], col=columns[2], row=columns[3],
                    data=df, kind='bar')
        st.pyplot(plt)


st.set_page_config(page_title='CheXpert Explorer')

ROW_COLUMNS = [cd.COL_SEX, cd.COL_AGE_GROUP, cd.COL_TRAIN_VALIDATION, cd.COL_FRONTAL_LATERAL]
rows = st.sidebar.multiselect('Select rows', ROW_COLUMNS)
columns = st.sidebar.multiselect('Select columns', ROW_COLUMNS)

labels = st.sidebar.multiselect('Show count of images with these labels (select one or more)',
                                get_labels(), default=ALL_LABELS)
# Warn the user that "all labels" is ignored when used with other labels
if ALL_LABELS in labels and len(labels) > 1:
    st.sidebar.write('Ignoring "{}" when used with other labels'.format(ALL_LABELS))

percentages = st.sidebar.radio('Add percentages across', ('Rows', 'Columns', 'No percentages'))

if not rows:
    st.write('Select rows and columns')
elif not set(rows).isdisjoint(columns):
    st.write('Rows and columns cannot have the same values')
else:
    # Remove the special "all labels" label if it's present in the list
    adjusted_labels = labels[:]  # make a copy
    if ALL_LABELS in adjusted_labels:
        adjusted_labels.remove(ALL_LABELS)

    totals = True
    df_agg = get_images_count(adjusted_labels, rows, columns, totals=totals,
                              percentages=percentages.lower(), format=True)
    df_agg = df_agg.copy()  # copy to avoid Streamlit caching warning
    if df_agg.empty:
        st.write('There are no images with this combination of filters.')
    else:
        st.table(df_agg)
        show_graph(get_images_count(adjusted_labels, rows, columns, totals=False))
