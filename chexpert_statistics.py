"""Calculate statistics for a CheXpert dataset.

It works on the DataFrame, not on the CheXPertDataset class, to allow statistics on filtered data.

Example::

    import chexpert_dataset as cxd
    import chexpert_statistics as cxs

    # Get a ChexPert dataset
    cxdata = cxd.CheXpertDataset()
    cxdata.fix_dataset() # optional

    # Calculate statistics on it
    stats = cxs.patient_study_image_count(cxdata.df)
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
COL_PERCENTAGE_CUMULATIVE = 'Cumulative %'
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


def _add_percentage(df: pd.DataFrame, level=0, cumulative=False) -> pd.DataFrame:
    """Add percentages to a multi-index DataFrame in long format, using the given index level."""
    # It must be a dataset in long format
    assert len(df.columns) == 1

    df[COL_PERCENTAGE] = df.groupby(level=level).apply(lambda x: 100 * x / x.sum())
    if(cumulative):
        df[COL_PERCENTAGE_CUMULATIVE] = df.groupby(level=level)[COL_PERCENTAGE].cumsum()
    return df


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

    if add_percentage:
        stats = _add_percentage(stats, level=1)

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


def images_per_patient_binned(df: pd.DataFrame, add_percentage: bool = True) -> pd.DataFrame:
    """Get the binned number of images per patient, split by training/validation set.

    Args:
        df (pd.DataFrame): A CheXpert DataFrame.
        add_percentage (bool, optional): Wheter to add percentage across sets. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame with the binned image counts, in long format (columns with the
            observation and a multiindex to identify set and bins).
    """
    stats = images_per_patient(df)
    bins = [0, 1, 2, 3, 10, 20, 100]
    bin_labels = ['1 image', '2 images', '3 images', '4 to 10 images', '11 to 20 images',
                  'More than 20 images']
    stats[IMAGES] = pd.cut(stats.Images, bins=bins, labels=bin_labels, right=True)
    summary = stats.reset_index().groupby([cxd.COL_TRAIN_VALIDATION, IMAGES], as_index=True,
                                          observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique))

    assert summary.loc[cxd.TRAINING].sum()[0] == cxd.PATIENT_NUM_TRAINING
    assert summary.loc[cxd.VALIDATION].sum()[0] == cxd.PATIENT_NUM_VALIDATION

    summary = _add_percentage(summary, level=0, cumulative=True)
    return summary


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


def main():
    """Test code to be invoked from the command line."""
    cxdata = cxd.CheXpertDataset()
    cxdata.fix_dataset()
    stats = patient_study_image_count(cxdata.df)
    print(stats)


if __name__ == '__main__':
    main()
