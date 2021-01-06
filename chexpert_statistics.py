"""Calculate statistics for a CheXpert dataset.

It works on the DataFrame, not on the CheXPertDataset class, to allow statistics on filtered data.

Example::

    import chexpert_dataset as cxd
    import chexpert_statistics as cxs

    # Get a ChexPert dataset
    cxdata = cxd.CheXpertDataset()
    cxdata.fix_dataset() # optional

    # Calculate statistics on it
    cxstats = cxs.ChexPer
    ...
"""
import pandas as pd
import chexpert_dataset as cxd

# Index names, index values, and column names for functions that return DataFrames with statistics
# Most statistics DataFrames are long (stacked) - these index names are combined into MultiIndex
# indices as needed
# Whenever possible, they match the name of the column used to group the statistics
INDEX_NAME_SET = 'Set'
INDEX_NAME_ITEM = 'Item'
COL_COUNT = 'Count'
COL_PERCENTAGE = '%'
PATIENTS = 'Patients'
IMAGES = 'Images'
STUDIES = 'Studies'


def _summary_stats_by_set(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate the summary statistics of a DataFrame that has counts."""
    summary = df.groupby([cxd.COL_TRAIN_VALIDATION], as_index=True, observed=True).agg(
        Minimum=(column, 'min'),
        Maximum=(column, 'max'),
        Median=(column, 'median'),
        Mean=(column, 'mean'),
        Std=(column, 'std'))
    idx = pd.MultiIndex.from_product([summary.index, [column]], names=[
        INDEX_NAME_SET, INDEX_NAME_ITEM])
    summary.index = idx
    return summary


def patient_study_image_count(df: pd.DataFrame, add_percentage: bool = False) -> pd.DataFrame:
    """Get count of patients, studies, and images, split by training/validation set.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.
        add_percentage (bool, optional): Wheter to add percentage across sets. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame with the counts, in long format (only one column, with the
            count, and a multiindex to identify set and item type).
    """
    # We need a column that is unique for patient and study
    COL_PATIENT_STUDY = 'Patient/Study'
    df = df.copy()  # preserve the caller's data
    df[COL_PATIENT_STUDY] = ['{}-{}'.format(p, s) for p, s in
                             zip(df[cxd.COL_PATIENT_ID], df[cxd.COL_STUDY_NUMBER])]

    stats = df.groupby([cxd.COL_TRAIN_VALIDATION], as_index=True, observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Studies=(COL_PATIENT_STUDY, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count')
    )

    assert stats[PATIENTS].sum() == cxd.PATIENT_NUM_TOTAL
    assert stats[STUDIES].sum() == cxd.STUDY_NUM_TOTAL
    assert stats[IMAGES].sum() == cxd.IMAGE_NUM_TOTAL

    stats = pd.DataFrame(stats.stack())
    stats = stats.rename(columns={0: COL_COUNT})
    stats.index.names = [INDEX_NAME_SET, INDEX_NAME_ITEM]

    def pct_for_item(one_set_one_item):
        # Divide this set/item by the sum of the same item type in all sets, e.g. all patients
        # for the training and validation sets - ignore the first level index and get all
        # items of the same type (second level index)
        # MultiIndex slicing: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html # noqa
        # and https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#using-slicers # noqa
        # in particular
        all_items_for_set = stats.loc[(slice(None), one_set_one_item.name[1]), :].sum()
        return 100 * one_set_one_item[COL_COUNT] / all_items_for_set[COL_COUNT].sum()

    if add_percentage:
        stats[COL_PERCENTAGE] = stats.apply(pct_for_item, axis='columns')

    return stats


def studies_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number of studies for each patient.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.

    Returns:
        pd.DataFrame: Number of studies each patient has for each set (training/validation).
    """
    # The same study number may shows up more than once for the same patient (a study that has
    # more than one image), thus we need the unique count of studies in this case
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_PATIENT_ID], as_index=True,
                       observed=True).agg(
        Studies=(cxd.COL_STUDY_NUMBER, pd.Series.nunique))
    return stats


def images_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number of images for each patient.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.

    Returns:
        pd.DataFrame: Number of images each patient has for each set (training/validation).
    """
    # Image (view) numbers may be repeated for the same patient (they are unique only within
    # each study), thus in this case we need the overall count and not unique count
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_PATIENT_ID], as_index=True,
                       observed=True).agg(
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    return stats


def studies_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for the number of studies per patient.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.

    Returns:
        pd.DataFrame: Summary statistics for number of studies per patient for each set
        (training/validation).
    """
    stats = studies_per_patient(df)
    summary = _summary_stats_by_set(stats, STUDIES)
    return summary


def images_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for the number of images per patient.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.

    Returns:
        pd.DataFrame: Summary statistics for number of images per patient for each set
        (training/validation).
    """
    stats = images_per_patient(df)
    summary = _summary_stats_by_set(stats, IMAGES)
    return summary
