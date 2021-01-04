"""Make it easier to work with the CheXpert dataset.

- Combine the training and validation sets into one DataFrame
- Create explicit columns for patient ID, study nubmer, and view number, instead of enconding in
  the path
- Add a column for age groups to help  cross-sectional analysis
- Adjust the column data types to reduce memory usage
- The "no mention" encoded as an empty string in the validation set is converted to an integer
  label for consistency and also to allow converting the columns to an integer, saving memory

Using from the command line: ``python3 -m preprocess > chexpert.csv``

From another module::

    import chexpert_dataset as cd
    chexpert = cd.CheXpert()
    chexpert.fix_dataset() # optional
    chexpert.df.head()

IMPORTANT: because we are using categories, set observed=True when using groupby with the
categorical columns to avoid surprises (https://github.com/pandas-dev/pandas/issues/17594)
"""
# pylint: disable=too-few-public-methods

import logging
import os
import re
import pandas as pd
import imagesize

# Dataset values that must hold when we manipulate it (groupby, pivot_table, filters, etc.)
IMAGE_NUM_TRAINING = 223_414
IMAGE_NUM_VALIDATION = 234
IMAGE_NUM_TOTAL = IMAGE_NUM_VALIDATION + IMAGE_NUM_TRAINING
PATIENT_NUM_TRAINING = 64_540
PATIENT_NUM_VALIDATION = 200
PATIENT_NUM_TOTAL = PATIENT_NUM_VALIDATION + PATIENT_NUM_TRAINING

# Labels
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
LABEL_UNCERTAIN = -1
LABEL_NO_MENTION = -99

# Observations (must match the names in the .csv files)
OBSERVATION_NO_FINDING = 'No Finding'
OBSERVATION_PATHOLOGY = sorted(['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                'Pneumothorax', 'Pleural Effusion', 'Pleural Other'])
OBSERVATION_OTHER = [OBSERVATION_NO_FINDING, 'Fracture', 'Support Devices']
OBSERVATION_ALL = OBSERVATION_OTHER + OBSERVATION_PATHOLOGY

# Names of some commonly-used columns already in the dataset
COL_SEX = 'Sex'
COL_AGE = 'Age'
COL_FRONTAL_LATERAL = 'Frontal/Lateral'
COL_AP_PA = 'AP/PA'

# Names of the columns added with this code
COL_PATIENT_ID = 'Patient ID'
COL_STUDY_NUMBER = 'Study Number'
COL_VIEW_NUMBER = 'View Number'
COL_AGE_GROUP = 'Age Group'
COL_TRAIN_VALIDATION = 'Training/Validation'

# Values of columns added with this code
TRAINING = 'Training'
VALIDATION = 'Validation'


class CheXpert:
    """An augmented version of the CheXPert dataset.

    Create one DataFrame that combines the train.csv and valid.csv files, then augments it. The
    combined DataFrame appends the following columns to the existing dataset columns:

    - Patient number
    - Study number
    - View number
    - Age group (MeSH age group - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/)
    - "Train" or "Validation" image

    It also normalizes the labels to 0, 1, and -1 by converting floating point labels to integer
    (e.g. 0.0 to 0) and by filling in empty label columns with 0.
    """

    def __init__(self, directory: str = None, add_image_size: bool = False, verbose: bool = True):
        """Populate the augmented dataset.

        Once the class is initialized, the augmented dataset is available as a Pandas DataFrame in
        the ``df`` class variable.

        Args:
            directory (str, optional): The directory where the dataset is saved, or ``None`` to
                search for the directory. Defaults to None.
            add_image_size (bool, optional): Add the image size (takes a few seconds). Defaults to
            False.
            verbose (bool, optional): Turn verbose logging on/off. Defaults to off.
        """
        self.__init_logger(verbose)
        self.__directory = directory
        self.__add_image_size = add_image_size

        self.__df = self.__get_augmented_chexpert()

    @property
    def df(self):
        """Return the DataFrame that contains the training and validation test sets.

        Make a copy before modifying it. This code does not return a copy to increase performace.
        """
        return self.__df

    @staticmethod
    def find_directory() -> str:
        """Determine the directory where the dataset is stored.

        There are two versions of the dataset, small and large. They are stored in
        CheXpert-v1.0-small and CheXpert-v1.0-large respectively. To make the code generic, this
        function finds out what version is installed.

        Note: assumes that 1) only one of the versions is installed and 2) that it is at the same
        level where this code is being executed.

        Returns:
            str: The name of the images directory or an empty string if it can't find one.
        """
        for entry in os.scandir('.'):
            if entry.is_dir() and re.match(r'CheXpert-v\d\.\d-', entry.name):
                return entry.name
        return ''

    def fix_dataset(self):
        """Fix issues with the dataset (in place).

        See code for what is fixed.
        """
        # There is one record with sex 'Unknown'. There is only one image for that patient, so we
        # don't have another record where the sex could be copied from. Change it to "Female"
        # (it doesn't matter much which sex we pick because it is one record out of 200,000+).
        self.df.loc[self.df.Sex == 'Unknown', ['Sex']] = 'Female'
        self.df.Sex.cat.remove_unused_categories()

    def __init_logger(self, verbose: bool):
        """Init the logger.

        Args:
            verbose (bool): Turn verbose logging on/off.
        """
        self.__ch = logging.StreamHandler()
        self.__ch.setFormatter(logging.Formatter('%(message)s'))
        self.__logger = logging.getLogger(__name__)
        self.__logger.addHandler(self._ch)
        self.__logger.setLevel(logging.INFO if verbose else logging.ERROR)

    def __get_augmented_chexpert(self) -> pd.DataFrame:
        """Get and augmented vresion of the CheXpert dataset.

        Add columns described in the file header and compacts the DataFrame to use less memory.

        Raises:
            RuntimeError: Cannot find the dataset directory and no directory was specified.

        Returns:
            pd.DataFrame: The dataset with the original and augmented columns.
        """
        directory = CheXpert.find_directory() if self.__directory is None else self.__directory
        if not directory:
            raise RuntimeError('Cannot find the CheXpert directory')
        self.__logger.info('Using the dataset in %s', directory)

        df = pd.concat(pd.read_csv(os.path.join(directory, f)) for f in ['train.csv', 'valid.csv'])

        # Convert the "no mention" label to an integer representation
        # IMPORTANT: assumes this is the only case of NaN after reading the .csv files
        self.__logger.info('Converting "no mention" to integer')
        df.fillna(LABEL_NO_MENTION, inplace=True)

        # Add the patient ID column by extracting it from the filename
        # Assume that the 'Path' column follows a well-defined format and extract from "patientNNN"
        self.__logger.info('Adding patient ID')
        df[COL_PATIENT_ID] = df.Path.apply(lambda x: int(x.split('/')[2][7:]))

        # Add the study number column, also assuming that the 'Path' column is well-defined
        self.__logger.info('Adding study number')
        df[COL_STUDY_NUMBER] = df.Path.apply(lambda x: int(x.split('/')[3][5:]))

        # Add the view number column, also assuming that the 'Path' column is well-defined
        self.__logger.info('Adding view number')
        view_regex = re.compile('/|_')
        df[COL_VIEW_NUMBER] = df.Path.apply(lambda x: int(re.split(view_regex, x)[4][4:]))

        # Add the MeSH age group column
        # Best reference I found for that: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1794003/
        # We have only complete years, so we can't use 'newborn'
        # Also prefix with zero because visualizers sort by ASCII code, not numeric value
        self.__logger.info('Adding age group')
        bins = [0, 2, 6, 13, 19, 45, 65, 80, 120]
        ages = ['(0-1) Infant', '(02-5) Preschool', '(06-12) Child', '(13-18) Adolescent',
                '(19-44) Adult', '(45-64) Middle age', '(65-79) Aged', '(80+) Aged 80']
        df[COL_AGE_GROUP] = pd.cut(df.Age, bins=bins, labels=ages, right=False)

        # Add the train/validation column
        self.__logger.info('Adding train/validation')
        df[COL_TRAIN_VALIDATION] = df.Path.apply(lambda x: TRAINING if 'train' in x else VALIDATION)

        # Add the image information column
        if self.__add_image_size:
            self.__logger.info('Adding image size (takes a few seconds)')
            size = [imagesize.get(f) for f in df.Path]
            df[['Width', 'Height']] = pd.DataFrame(size, index=df.index)

        # Optimize memory usage: use categorical values and small integer when possible
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html
        # IMPORTANT: because we are using categories, set observed=True when using groupby with
        # these columns to avoid surprises (https://github.com/pandas-dev/pandas/issues/17594)
        for c in [COL_SEX, COL_FRONTAL_LATERAL, COL_AP_PA, COL_AGE_GROUP, COL_TRAIN_VALIDATION]:
            df[c] = df[c].astype('category')
        for c in [COL_AGE, COL_PATIENT_ID, COL_STUDY_NUMBER, COL_VIEW_NUMBER]:
            df[c] = df[c].astype('int32')
        for c in OBSERVATION_ALL:
            df[c] = df[c].astype('int8')

        # A bare minimum amount of sanity checks
        assert df[df[COL_TRAIN_VALIDATION] ==
                  TRAINING][COL_PATIENT_ID].nunique() == PATIENT_NUM_TRAINING
        assert df[COL_PATIENT_ID].nunique() == PATIENT_NUM_TOTAL

        return df


def main():
    """Separate main function to follow conventions and docstring to make pylint happy."""
    chexpert = CheXpert()
    chexpert.fix_dataset()
    print(chexpert.df.to_csv(index=False))


if __name__ == '__main__':
    main()
