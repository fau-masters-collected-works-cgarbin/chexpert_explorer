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

The functions take as input a DataFrame create by a CheXpertDataset class and return the requested
statistics/summary. The DataFrame may be filtered (e.g. only the training rows), as long as it has
the same number of columns as the original CheXPertDataset DataFrame.

The returned DataFrames are in long format, i.e. one observation per row.
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
COL_PATIENT_STUDY = 'Patient/Study'
COL_AGED = 'Aged'

COL_PERCENTAGE = '%'
COL_PERCENTAGE_CUMULATIVE = 'Cum. %'
COL_PERCENTAGE_PATIENTS = 'Patients %'
COL_PERCENTAGE_IMAGES = 'Images %'

COL_LABEL_POSITIVE = 'Positive'
COL_LABEL_NEGATIVE = 'Negative'
COL_LABEL_UNCERTAIN = 'Uncertain'
COL_LABEL_NO_MENTION = 'No mention'

PATIENTS = 'Patients'
IMAGES = 'Images'
STUDIES = 'Studies'


def _pct(x): return 100 * x / x.sum()


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

    df[COL_PERCENTAGE] = df.groupby(level=level).apply(_pct)
    if cumulative:
        df[COL_PERCENTAGE_CUMULATIVE] = df.groupby(level=level)[COL_PERCENTAGE].cumsum()
    return df


def _add_aux_patient_study_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column that is unique for patient/study to help with grouping and counting."""
    df = df.copy()  # preserve the caller's data
    df[COL_PATIENT_STUDY] = ['{}-{}'.format(p, s) for p, s in
                             zip(df[cxd.COL_PATIENT_ID], df[cxd.COL_STUDY_NUMBER])]
    return df


def patient_study_image_count(df: pd.DataFrame, add_percentage: bool = False,
                              filtered: bool = False) -> pd.DataFrame:
    """Calculate count of patients, studies, and images, split by training/validation set."""
    df = _add_aux_patient_study_column(df)

    stats = df.groupby([cxd.COL_TRAIN_VALIDATION], as_index=True, observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Studies=(COL_PATIENT_STUDY, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))

    # Validate against expected CheXpert number when the DataFrame is not filtered
    if not filtered:
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
    """Calculate the number of studies for each patient."""
    # The same study number may shows up more than once for the same patient (a study that has
    # more than one image), thus we need the unique count of studies in this case
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_PATIENT_ID], as_index=True,
                       observed=True).agg(
        Studies=(cxd.COL_STUDY_NUMBER, pd.Series.nunique))
    return stats


def images_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number of images for each patient."""
    # Image (view) numbers may be repeated for the same patient (they are unique only within
    # each study), thus in this case we need the overall count and not unique count
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_PATIENT_ID], as_index=True,
                       observed=True).agg(
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    return stats


def images_per_patient_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number of images for each patient, split by sex."""
    # Image (view) numbers may be repeated for the same patient (they are unique only within
    # each study), thus in this case we need the overall count and not unique count
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_SEX], as_index=True, observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))

    stats[COL_PERCENTAGE_PATIENTS] = stats.groupby(level=0)[PATIENTS].apply(_pct)
    stats[COL_PERCENTAGE_IMAGES] = stats.groupby(level=0)[IMAGES].apply(_pct)

    # Adjust the column order to a logical sequence
    stats = stats[[PATIENTS, COL_PERCENTAGE_PATIENTS, IMAGES, COL_PERCENTAGE_IMAGES]]

    # Improve presentation of the columns headers
    columns = pd.MultiIndex.from_product([[PATIENTS, IMAGES],
                                          [COL_COUNT, COL_PERCENTAGE]])
    stats.columns = columns

    return stats


def images_per_patient_binned(df: pd.DataFrame, filtered: bool = False) -> pd.DataFrame:
    """Calculate the binned number of images per patient, split by training/validation set."""
    stats = images_per_patient(df)
    bins = [0, 1, 2, 3, 4, 5, 10, 20, 30, 100]
    bin_labels = ['1 image', '2 images', '3 images', '4 images', '5 images', '6 to 10 images',
                  '11 to 20 images', '21 to 30 images', 'More than 30 images']
    num_images = 'Number of images'
    stats[num_images] = pd.cut(stats.Images, bins=bins, labels=bin_labels, right=True)
    group = stats.reset_index().groupby([cxd.COL_TRAIN_VALIDATION, num_images], as_index=True,
                                        observed=True)

    patient_summary = group.agg(Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique))
    if not filtered:
        assert patient_summary.loc[cxd.TRAINING].sum()[0] == cxd.PATIENT_NUM_TRAINING
        assert patient_summary.loc[cxd.VALIDATION].sum()[0] == cxd.PATIENT_NUM_VALIDATION
    patient_summary = _add_percentage(patient_summary, level=0, cumulative=True)

    image_summary = group.agg(Images=(IMAGES, 'sum'))
    if not filtered:
        assert image_summary.loc[cxd.TRAINING].sum()[0] == cxd.IMAGE_NUM_TRAINING
        assert image_summary.loc[cxd.VALIDATION].sum()[0] == cxd.IMAGE_NUM_VALIDATION
    image_summary = _add_percentage(image_summary, level=0, cumulative=True)

    summary = patient_summary.join(image_summary, lsuffix=' ' + PATIENTS, rsuffix=' ' + IMAGES)

    columns = pd.MultiIndex.from_product([[PATIENTS, IMAGES],
                                          [COL_COUNT, COL_PERCENTAGE, COL_PERCENTAGE_CUMULATIVE]])
    summary.columns = columns

    return summary


def studies_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for the number of studies per patient."""
    stats = studies_per_patient(df)
    summary = _summary_stats_by_set(stats, STUDIES)
    return summary


def images_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for the number of images per patient."""
    stats = images_per_patient(df)
    summary = _summary_stats_by_set(stats, IMAGES)
    return summary


def label_image_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number and percentage of times each observation appears in images."""
    images_in_set = len(df[cxd.COL_VIEW_NUMBER])
    observations = cxd.OBSERVATION_OTHER + cxd.OBSERVATION_PATHOLOGY
    all_labels = [cxd.LABEL_POSITIVE, cxd.LABEL_NEGATIVE, cxd.LABEL_UNCERTAIN, cxd.LABEL_NO_MENTION]
    col_names = [COL_LABEL_POSITIVE, COL_PERCENTAGE, COL_LABEL_NEGATIVE, COL_PERCENTAGE,
                 COL_LABEL_UNCERTAIN, COL_PERCENTAGE, COL_LABEL_NO_MENTION, COL_PERCENTAGE]
    stats = pd.DataFrame(index=observations, columns=col_names)
    for obs in observations:
        count = [len(df[df[obs] == x]) for x in all_labels]
        pct = [c*100/images_in_set for c in count]
        # Interleave count and percentage columns
        stats.loc[obs] = [x for t in zip(count, pct) for x in t]
    # Sanity check: check a few columns for the number of images
    cols_with_counts = [v for v in col_names if v != '%']
    assert stats.loc[cxd.OBSERVATION_NO_FINDING][cols_with_counts].sum() == images_in_set
    assert stats.loc[cxd.OBSERVATION_PATHOLOGY[1]][cols_with_counts].sum() == images_in_set
    return stats


def observation_image_coincidence(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate how many times each observation appears with (is positive with) another one."""
    labels = cxd.OBSERVATION_OTHER + cxd.OBSERVATION_PATHOLOGY
    stats = pd.DataFrame(index=labels, columns=labels)

    for label in labels:
        df_label = df[df[label] == 1]
        coincidences = [len(df_label[df_label[other_label] == 1]) for other_label in labels]
        stats.loc[label] = coincidences
    # Sanity check: 'No Finding' should not coincide with a pathology
    assert stats.loc[cxd.OBSERVATION_NO_FINDING][cxd.OBSERVATION_PATHOLOGY].sum() == 0
    return stats


def patients_images_by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate number of patients and images by age group."""
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP], as_index=True,
                       observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    return stats


def patients_images_by_sex_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate number of patients and images by sex and age group."""
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP, cxd.COL_SEX], as_index=True,
                       observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))

    assert stats[PATIENTS].sum() == cxd.PATIENT_NUM_TOTAL_BY_AGE_GROUP
    assert stats[IMAGES].sum() == cxd.IMAGE_NUM_TOTAL

    return stats


def patients_studies_images_by_sex_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate number of patients, studies and images by sex and age group."""
    df = _add_aux_patient_study_column(df)
    stats = df.groupby([cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP, cxd.COL_SEX], as_index=True,
                       observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique),
        Studies=(COL_PATIENT_STUDY, pd.Series.nunique),
        Images=(cxd.COL_VIEW_NUMBER, 'count'))

    assert stats[PATIENTS].sum() == cxd.PATIENT_NUM_TOTAL_BY_AGE_GROUP
    assert stats[STUDIES].sum() == cxd.STUDY_NUM_TOTAL
    assert stats[IMAGES].sum() == cxd.IMAGE_NUM_TOTAL

    return stats


def patients_studies_images_by_sex_age_group_subtotal(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate number of patients, studies and images by sex and age group.

    This is the same data as the function above, adding subtotals for each category. The table is
    already in wide format because of the subtotal columns.
    """
    # We rely on the logic to build subgroups above
    # We call that function just to check if the logic is correct, via the "assert" lines there
    patients_studies_images_by_sex_age_group(df)

    df = _add_aux_patient_study_column(df)
    cols = [cxd.COL_TRAIN_VALIDATION, cxd.COL_AGE_GROUP, cxd.COL_SEX]
    sum_index = ('',  'All')

    patients = df.groupby(cols, as_index=True, observed=True).agg(
        Patients=(cxd.COL_PATIENT_ID, pd.Series.nunique))
    patients = patients.unstack(fill_value=0)
    patients[sum_index] = patients.sum(axis=1)

    studies = df.groupby(cols, as_index=True, observed=True).agg(
        Studies=(COL_PATIENT_STUDY, pd.Series.nunique))
    studies = studies.unstack(fill_value=0)
    studies[sum_index] = studies.sum(axis=1)

    images = df.groupby(cols, as_index=True, observed=True).agg(
        Images=(cxd.COL_VIEW_NUMBER, 'count'))
    images = images.unstack(fill_value=0)
    images[sum_index] = images.sum(axis=1)

    stats = pd.concat([patients, studies, images], axis=1)
    return stats


def aged_patients(df: pd.DataFrame) -> pd.DataFrame:
    """List of patients that aged during the timeframe of the dataset.

    Aged patients are the ones that have studies at different ages that cross age groups. This
    explains why when we group the dataset by age group and count the number of patients we end up
    with more patients than when we count only by patient ID. An aged patient will have multiple
    rows when we group by "patient ID + age group".
    """
    # The same study number may shows up more than once for the same patient (a study that has
    # more than one image), thus we need the unique count of studies in this case
    stats = df.groupby([cxd.COL_PATIENT_ID, cxd.COL_AGE_GROUP], as_index=False,
                       observed=True).agg(
        Studies=(cxd.COL_STUDY_NUMBER, pd.Series.nunique))
    stats[COL_AGED] = stats[cxd.COL_PATIENT_ID].diff().eq(0)

    # Number of patients when we count their age groups separately
    # print(stats.shape[0])
    # Difference between the number above and the count of patients without considering age groups
    # print(stats.shape[0] - cxd.PATIENT_NUM_TOTAL)
    # One of the patients (that crossed multiple age groups)
    # print(df[df[cxd.COL_PATIENT_ID] == 23])

    return stats[stats[COL_AGED]]


def main():
    """Test code to be invoked from the command line."""
    cxdata = cxd.CheXpertDataset()
    cxdata.fix_dataset()
    stats = patients_studies_images_by_sex_age_group_subtotal(cxdata.df)
    print(stats)


if __name__ == '__main__':
    main()
