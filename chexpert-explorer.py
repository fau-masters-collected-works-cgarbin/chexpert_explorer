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
TOTAL = 'Total'


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
def get_images_count(labels: List[str], rows: List[str], columns: List[str],
                     totals: bool = False) -> pd.DataFrame:
    """Get a pivot table with the image count for the select filters.

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
                         margins=totals, margins_name=TOTAL, dropna=False)
    pvt.fillna(0, inplace=True)

    return pvt


@st.cache
def merge_percentages(df: pd.DataFrame, pct_by: str, total_col_name: str = None) -> pd.DataFrame:
    """Add percentages to a DataFrame that has counters.

    Args:
        df (pd.DataFrame): The DataFrame that has counters only.
        pct_by (str): Whether to calculates percentages by ``rows`` or ``columns``.
        total_col_name (str, optional): The name of a "totals" column, if there is one. Used to
            adjust the percentage calculation (avoids double counting). Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with a percentage column next to each counter column.
    """
    # Start with two dataframes, one with the counter and another with the percentage that have only
    # only column (the counter or the percentage) - everything else is an index
    df_flat = get_flattened_df(df, reset_index=False)
    pct_flat = get_flattened_df(get_percentages(
        df, total_col_name is not None, pct_by), reset_index=False)

    # Merge them into one dataframe that has the two columns, counter and percentage
    # We start with a copy to not mess with the original version (it may be cached, e.g. when using
    # with Streamlit)
    df_combined = df_flat.copy()
    df_combined['%'] = pct_flat[pct_flat.columns[-1]]

    # With the counter and percentage columns in place, we can move the original columns back to
    # their place ("unflat)")
    # This restores the layout of the original dataframe, with the percentage columns all the way
    # to the right
    unstack_indices = list(range(df.index.nlevels, df_flat.index.nlevels))
    df_combined = df_combined.unstack(unstack_indices, fill_value=0)

    # Interleave the percentage columns (to the right) with the counter columns (to the left), so
    # that each counter column is followed by its respective percentage column
    all_cols = df_combined.columns.values
    counter_cols = all_cols[:int(len(all_cols)/2)]
    pct_cols = all_cols[int(len(all_cols)/2):]
    interleaved_cols = [c for pair in zip(counter_cols, pct_cols) for c in pair]
    df_combined = df_combined[interleaved_cols]

    # At this point, the image/percentage column is at the top of the multiindex
    # Move it to the bottom of the multiindex
    top_moved_to_bottom = list(range(1, df_combined.columns.nlevels))
    top_moved_to_bottom.append(0)
    df_combined = df_combined.reorder_levels(top_moved_to_bottom, axis='columns')

    # Sort the columns to display them in a logical way
    df_combined.sort_index(axis='columns', level=0, sort_remaining=False, inplace=True)

    # If there is a total column, make sure it's the last one
    if total_col_name is not None:
        top_level_columns = list(df_combined.columns.get_level_values(0).unique())
        if total_col_name in top_level_columns:
            top_level_columns.remove(total_col_name)
            top_level_columns.append(total_col_name)
            df_combined = df_combined.reindex(top_level_columns, axis='columns', level=0)

    return df_combined


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
def get_flattened_df(df_agg: pd.DataFrame, reset_index=False) -> pd.DataFrame:
    """Get a flattened version of the DataFrame.

    The flattened version of the DataFrames has only one column with values (the count of images).
    The other columns in the DataFrame are descriptions of that value.

    Args:
        df_agg (pd.DataFrame): The aggregrated DataFrame, created with pivot_table or groupby.
        reset_index: Whether to reset the index, i.e. turn all indices into columns, or leave the
            indices in place, as created with the stacking operation.

    Returns:
        pd.DataFrame: The flattened DataFrame
    """
    # Flatten the dataframe
    df = df_agg.stack(list(range(df_agg.columns.nlevels)))
    if reset_index:
        df = df.reset_index()
    else:
        df = pd.DataFrame(df)
    # Rename the count column to something more meaningful
    df.rename(columns={0: 'Images'}, inplace=True)
    return df


def show_graph(df_agg: pd.DataFrame):
    """Show a graph for the aggregated DataFrame.

    The graph is a collection of bar plots, categorized by the columns, left to right.  The second
    column is always used as the hue. The other columns are used for rows and columns in the graph.

    Args:
        df_agg (pd.DataFrame): The aggregrated DataFrame to graph.
    """
    df = get_flattened_df(df_agg, reset_index=True)

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

    totals = True
    df_agg = get_images_count(adjusted_labels, rows, columns, totals=totals)
    if df_agg.empty:
        st.write('There are no images with this combination of filters.')
    else:
        pct_lower = percentages.lower()
        if pct_lower in ['rows', 'columns']:
            df_agg = merge_percentages(df_agg, pct_lower, TOTAL)

        # Streamlit does not yet support formatting for multiindex dataframes
        # It will be added with https://github.com/streamlit/streamlit/issues/956
        # Until that issue is resolved, we need to format the numbers ourselves
        df_agg = df_agg.applymap(lambda x: '{:,.0f}'.format(x))

        st.table(df_agg)
        show_graph(get_images_count(adjusted_labels, rows, columns, totals=False))
