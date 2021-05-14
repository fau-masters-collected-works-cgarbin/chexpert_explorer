# CheXpert preprocessing and visualization

A preprocessor and explorer for [CheXPert](https://stanfordmlgroup.github.io/competitions/chexpert/).

This code was used to generate the tables for the [CheXpert datasheet](https://arxiv.org/abs/2105.03020).

## Preparing the environment

If this is the first time you are using the preprocessor and the explorer, follow
[the setup instructions](./setup.md) to prepare the environment.

## Preprocessing and extracting statistics

`chexpert_dataset.py` reads the two .csv files from the dataset and converts them into one DataFrame
that simplifies exploring and filtering the dataset.

- Create explicit columns for patient ID, study number, and view number, extracted from the file paths.
- Adjust the column data types to reduce memory usage.
- Add a column for age groups to help cross-sectional analysis.
- Labels are encoded as integers, including the "no mention" (encoded as an empty string in the validation
  set is converted for consistency.

Once we have the dataset in this format, we can use Pandas and NumPy to extract statistics. The module
`chexpert_statistics.py` has several functions already in place.

Example (from `sample_code.py`):

```python
import chexpert_dataset as cxd
import chexpert_statistics as cxs

cxdata = cxd.CheXpertDataset()

# Count of patients and images in the training and validation datasets
# Long format, easier to index in code
stats = cxs.patient_study_image_count(cxdata.df)
print(stats)

# Wide format, better for documentation
stats = stats.unstack().droplevel(0, axis='columns')
print(stats)
```

## Visual exploration with Streamlit

Run [Streamlit](https://www.streamlit.io/) to explore the dataset by finding.

`streamlit run chexpert_explorer.py`

Streamlit opens a web page where you can select filters to explore the dataset.

The dataset is large. Applying some of the filters may take several seconds.

## LaTeX exporter

`chexpert_latex_export.py` exports the dataset summary and cross-sectional
statistics as LaTeX tables.

Used to create tables for the [CheXpert datasheet](https://arxiv.org/abs/2105.03020)
programmatically, reducing the chance of errors.

NOTE: a variable at the top of the file defines the path to export the tables to. Adjust it as
needed for your environment.

To generate the tables:

```bash
python3 chexpert_latex_export.py
```

## Invariants to check the code

`chexpert_dataset.py` defines several constants about the dataset:

```python
PATIENT_NUM_TRAINING = 64_540
PATIENT_NUM_VALIDATION = 200
PATIENT_NUM_TOTAL = PATIENT_NUM_VALIDATION + PATIENT_NUM_TRAINING
STUDY_NUM_TRAINING = 187_641
STUDY_NUM_VALIDATION = 200
STUDY_NUM_TOTAL = STUDY_NUM_TRAINING + STUDY_NUM_VALIDATION
IMAGE_NUM_TRAINING = 223_414
IMAGE_NUM_VALIDATION = 234
IMAGE_NUM_TOTAL = IMAGE_NUM_VALIDATION + IMAGE_NUM_TRAINING
```

These constants can be used when generating summary statistics to ensure that while grouping
and summarizing the dataset we are not making a mistake that results in incorrect values.

For example, the function used in the sample code is protected with `assert` statements
as shown below.

```python
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
    ...
```
