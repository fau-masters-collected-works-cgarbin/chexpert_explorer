# CheXpert preprocessing and visualization

A preprocessor and visualizer for [CheXPert](https://stanfordmlgroup.github.io/competitions/chexpert/).

- Preprocessing: creates a single .csv suitable to explore the dataset with spreadsheet programs
  (Excel, Google Sheets, etc.).
- Visualizer: explores the images in the dataset.

## Preparing to use the tools

### Download the dataset

The license terms for CheXPert does not allow redistribution (understandably). Before using the
preprocessor and the visualizer, you need to download your own copy of the dataset and uncompress
it into this project's folder.

1. Go to the [CheXPert page](https://stanfordmlgroup.github.io/competitions/chexpert/) and request
   the dataset (scroll to the bottom of that page).
1. Download the .zip file and uncompress it inside the main directory of this project.

Once you are done you should have a directory strcuture that looks like this:

```text
   This project's main directory
    |- README.md (this file)
    |- ... other project files and folders
    +- CheXpert-v1.0-small
       |- train.csv
       |- valid.csv
       |- train
       |  |- patientXXXX (several directories)
       |- valid
       |  |- patientXXXX (several directories)
```

### Prepare the Python environment

TBD

## Preprocessing

Why do we need to preprocess the dataset? The dataset description is delivered as two files, one for
the training (train.csv) and one for the validation set (valid.csv). They also have slightly
different representations for the labels (floating points in the test set and integers in the
validation set).

Preprocessing the dataset makes it easier to explore it with spreadsheet programs, e.g. create
filter and pivot tables.

The preprocessor creates one .csv file that combines the train.csv and valid.csv files. Compared to
the original dataset files, the combined file has:

- One column to indicate if the image is from the training or validation set.
- Labels normalized to integers.

## Visualizing

TBD
