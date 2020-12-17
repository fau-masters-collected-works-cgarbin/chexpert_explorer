# Preparing to use the tools

Before using the preprocessor and the visualizer for the first time you need to:

1. Download the dataset
1. Prepare the Python environment

Follow the directions in the next sections to put that in place. They have to be done only once.

## Download the dataset

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

## Prepare the Python environment

- Install Python 3
- Go into this repository's directory
- Create a Python [environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
  `python3 -m venv env`
- Activate the environmnet: `source env/bin/activate` (Linux/Mac) or `.\env\Scripts\activate` (Windows)
- Install the Python packages: `pip install -r requirements.txt`
