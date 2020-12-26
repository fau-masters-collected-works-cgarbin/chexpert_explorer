"""DataFrame utility functions."""

import pandas as pd


def get_percentages(df: pd.DataFrame, totals: bool, percentages: str) -> pd.DataFrame:
    """Return a percentage version of the DataFrame, across rows or columns, as specified.

    Based on https://stackoverflow.com/a/42006745.

    Args:
        df (pd.DataFrame): The DataFrame with the raw numbers.
        totals (bool): Whether the DataFrame has a "totals" row and column (when aggregated with
            "margins" set to True).
        percentages (str): Whether to calculate percentages by ``rows`` or ``columns``.

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
        # Stacking may result in a series - make sure we return a DataFrame
        df = pd.DataFrame(df)
    # Rename the count column to something more meaningful
    df.rename(columns={0: 'Images'}, inplace=True)
    return df


def merge_percentages(df: pd.DataFrame, pct_by: str, total_col_name: str = None) -> pd.DataFrame:
    """Add percentages to a DataFrame that has counters.

    Args:
        df (pd.DataFrame): The DataFrame that has a column with counters only (everything else is
            an index)
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
