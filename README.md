# Credit Default Risk Modeling

This repository contains a credit-default-risk modeling project based on the Kaggle **Home Credit Default Risk** competition. The objective is to estimate the probability that a loan applicant will have difficulty repaying a loan.

- `TARGET = 0`: the loan was repaid
- `TARGET = 1`: the client experienced repayment difficulty
- Main evaluation metric: **ROC AUC**

## Project history

I originally completed this project independently during an OpenClassrooms AI refresher course.

The original version placed most of the workflow in one notebook, with a large function handling much of the data preparation. That structure made the project difficult to debug and maintain. I revisited the project in **August 2026** to:

- separate data preparation, visualization, and modeling into dedicated notebooks;
- move reusable functions into focused Python modules;
- make the execution flow easier to inspect and debug;
- fix compatibility issues caused by changes in pandas and related libraries;
- preserve the original analysis and modeling approach while making the project easier to run.

The current update is primarily a **refactoring and compatibility pass**, not a complete redesign of the original project.

## Project structure

```text
.
├── data/                               # Local Kaggle data; not committed
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_visualization.ipynb
│   └── 03_model_training_evaluation.ipynb
├── src/
│   ├── constants.py
│   ├── data_io.py
│   ├── dataset_builder.py
│   ├── feature_engineering.py
│   ├── functions.py
│   ├── modeling.py
│   ├── outliers.py
│   ├── related_tables.py
│   └── visualization.py
├── .gitignore
└── README.md
```

## Main workflow

### 1. Data preparation

The first notebook:

- converts the original Kaggle CSV files to local pickle files;
- combines the application training and test tables;
- handles selected invalid and missing values;
- creates age, income, annuity, employment, and polynomial features;
- one-hot encodes categorical variables;
- aggregates the related Home Credit tables;
- creates the final merged modeling dataset.

### 2. Data visualization

The second notebook explores:

- target-class imbalance;
- missing values;
- numerical and categorical feature distributions;
- external credit-score features;
- engineered ratios and age groups;
- correlations between selected variables and the target.

### 3. Model training and evaluation

The third notebook includes:

- stratified train/test splitting and cross-validation;
- preprocessing pipelines;
- Logistic Regression, Random Forest, and Histogram Gradient Boosting;
- SMOTE for class imbalance;
- RFECV feature selection;
- GridSearchCV and RandomizedSearchCV;
- confusion matrices, classification reports, and ROC AUC;
- permutation importance and SHAP-based interpretation.

## Getting the Kaggle data

The data is **not included in this repository**. The competition files are subject to Kaggle's competition rules and are several gigabytes in total.

Competition page:

<https://www.kaggle.com/competitions/home-credit-default-risk/data>

Before downloading, sign in to Kaggle and accept the competition rules.

### Option A — Download through the Kaggle website

1. Open the competition data page.
2. Sign in and accept the competition rules.
3. Select **Download All**.
4. Extract the downloaded archive.
5. Copy the CSV files directly into this repository's `data/` directory.

The directory should contain at least:

```text
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── credit_card_balance.csv
├── installments_payments.csv
├── POS_CASH_balance.csv
└── previous_application.csv
```

### Option B — Download with the Kaggle API

Install the official Kaggle command-line tool:

```bash
python -m pip install kaggle
```

From your Kaggle account settings, create and download an API token named `kaggle.json`.

Place it in the Kaggle configuration directory:

- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- macOS/Linux: `~/.kaggle/kaggle.json`

Never commit `kaggle.json` to GitHub.

From the project root, download the competition archive into `data/`:

```bash
kaggle competitions download -c home-credit-default-risk -p data
```

Extract it using Python:

```bash
python -m zipfile -e data/home-credit-default-risk.zip data
```

You can then delete the downloaded ZIP archive.

## Installation

Create a virtual environment, activate it, and install the main dependencies:

```bash
python -m pip install \
  jupyter pandas numpy scipy scikit-learn imbalanced-learn \
  matplotlib seaborn missingno shap
```

## Running the project

Run the notebooks in this order:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_data_visualization.ipynb`
3. `notebooks/03_model_training_evaluation.ipynb`

Before the first run, enable the CSV conversion call in the setup cell of the data-preparation notebook:

```python
csv_to_pkl()
```

This creates the local pickle files expected by the remaining preparation steps. The notebooks also generate files such as:

```text
data/application_visualization.pkl
data/application_prepared.pkl
data/merged_dataset.pkl
```

These generated files are intentionally excluded from Git.

## Reproducibility and debug mode

All randomized operations use the shared constant:

```python
RND_SEED = 7
```

This includes train/test splitting, stratified cross-validation, SMOTE, resampling, randomized search, permutation importance, and models that expose a `random_state` parameter.

The model-training notebook keeps `debug = True` by default and evaluates a stratified 20% subset during expensive cross-validation and search steps. This makes the workflow practical on a local computer while preserving the target-class distribution. Set it to `False` only when enough memory and processing time are available. Metrics produced in debug mode should be described as subset results rather than full-dataset results.

## Notes

- The complete Home Credit dataset is large, so data preparation and model selection can require substantial memory and processing time.
- Use the existing debug or reduced-data options when testing the workflow before running it on the complete dataset.
- Data files, generated pickle files, reports, presentations, and Kaggle credentials are excluded through `.gitignore`.

## Author

**Thomas Runser**  
MSc in Artificial Intelligence, Pattern Recognition and Robotics  
GitHub: <https://github.com/ThomasRunser>
