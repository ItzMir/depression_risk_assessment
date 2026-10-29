# Depression Risk Assessment from Social Media Posts

Post-level binary classification of text posts as:

- `0 = Non-depressed`
- `1 = Depressed`

This is a **risk assessment system**, not a clinical diagnostic tool.

## Project principles

- The dataset's own label is the supervised target. Labels are never
  generated from engineered features.
- User-level splitting prevents leakage: a user appears in exactly one of
  TRAIN or FINAL TEST, and in exactly one CV validation fold.
- Four custom models are implemented from scratch:
  Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes.
- Forbidden libraries: `pandas`, `scikit-learn`, `scipy`, `nltk`, `spacy`,
  `tensorflow`, `pytorch`, `transformers`.
- Only `openpyxl` is used, and only for Excel documentation.

## Directory layout

    config/       configuration
    data/raw/     raw dataset (read-only)
    data/processed/  saved splits
    lexicons/     word lists
    src/          source code
    outputs/      models, predictions, metrics, excel, logs

## Setup

    pip install -r requirements.txt

Place the Kaggle CSV at `data/raw/twitter_depression.csv`.

## Running

From the project root:

    python -m src.main --model logistic
    python -m src.main --model tree
    python -m src.main --model forest
    python -m src.main --model nb

Each run handles loading, validation, feature extraction, splitting,
5-fold CV, final training, and final test evaluation for that one model.

## Outputs

- `outputs/excel/master_results.xlsx` — full documentation workbook
- `outputs/predictions/` — per-model final predictions
- `outputs/metrics/` — per-model metric CSVs
- `outputs/models/` — saved model parameters
- `outputs/logs/` — training, prediction, and error logs