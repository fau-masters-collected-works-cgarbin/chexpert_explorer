# CheXpert preprocessing and visualization

A preprocessor and visualizer for [CheXPert](https://stanfordmlgroup.github.io/competitions/chexpert/).

- Preprocessing: creates a single .csv suitable to explore the dataset with spreadsheet applications
  (Excel, Google Sheets, etc.).
- Visualizer: explores the images in the dataset.

## Preparing to use the tools

If this is the first time you are using the preprocessor and the visualizer, follow
[the setup instructions](./setup.md) to prepare the environment.

## Preprocessing the dataset

Why do we need to preprocess the dataset? The dataset description is delivered as two files, one for
the training (train.csv) and one for the validation set (valid.csv). They also have slightly
different representations for the labels (floating points in the test set and integers in the
validation set).

Preprocessing the dataset makes it easier to explore it with spreadsheet applications, e.g. create
filters, pivot tables, and graphs.

The preprocessor creates one .csv file that combines the train.csv and valid.csv files. Compared to
the original dataset files, the combined file has:

- One column to indicate if the image is from the training or validation set.
- Labels normalized to integers.

How to preprocess the dataset:

1. Run ... TBD
2. Open the .csv file in your favority spreadsheet application.

## Visualizing the dataset

TBD
