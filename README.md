# CheXpert preprocessing and visualization

A preprocessor and explorer for [CheXPert](https://stanfordmlgroup.github.io/competitions/chexpert/).

- Preprocessing: parses the .csv files that describe the dataset and augments it with more granular
  information. Results are stored as a Pandas DataFrame for easy filtering and summarization.
- Explorer: filters and explores the composition of the dataset by finding, using [Streamlit](https://www.streamlit.io/).
- Statistics generation: generates Pandas tables with different cross-sectional views into the dataset.
- LaTeX exporter: exports tables in LaTeX format. Used to populate the tables in the CheXpert
  datasheet programmatically.

## Preparing the environment

If this is the first time you are using the preprocessor and the explorer, follow
[the setup instructions](./setup.md) to prepare the environment.

## Preprocessing the dataset

Why do we need to preprocess the dataset? The dataset description is delivered as two files, one for
the training (train.csv) and one for the validation set (valid.csv). They also have slightly
different representations for the labels (floating points in the test set and integers in the
validation set).

Preprocessing the dataset makes it easier to explore it with spreadsheet applications, e.g. create
filters, pivot tables, and graphs.

The preprocessor creates a Pandas DataFrame that combines the training and validation .csv files,
augmenting them with more granular data.

To use it in code:

```python
import chexpert_dataset as cxd

cxdata = cxd.CheXpertDataset()

# Optionally, fix a few minor items in the dataset
cxdata.fix_dataset()

# Get the Pandas DataFrame
df = cxdata.df
```

To export the preprocessed data as CSV:

```bash
python3 chexpert_dataset.py
```

## Exploring the dataset by finding

Run [Streamlit](https://www.streamlit.io/) to explore the dataset by finding.

`streamlit run chexpert_explorer.py`

Streamlit opens a web page where you can select filters to explore the dataset.

The dataset is large. Applying some of the filters may take several seconds.

## Statistics generation

Use in combination with the dataset preprocessor to get cross-sectional and summary statistics for
the dataset.

```python
import chexpert_dataset as cxd
import chexpert_statistics as cxs

chexpert = cxd.CheXpertDataset()
chexpert.fix_dataset()
# Make code a bit simpler
df = chexpert.df

# Count of patients and images in the training and validation datasets
stats = cxs.patient_study_image_count(df)
stats = stats.unstack().droplevel(0, axis='columns')
```

## LaTeX exporter

Exports the dataset summary and cross-sectional statistics as LaTeX tables.

Used to create tables for the datasheet paper programmatically, reducing the chance of errors.

NOTE: a variable at the top of the file defines the path to export the tables to. Adjust it as
needed for your environment.

To generate the tables:

```bash
python3 chexpert_latex_export.py
```
