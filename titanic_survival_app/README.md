# Titanic Survival Predictor

A beginner-friendly Streamlit application that estimates the survival outcome
of a Titanic passenger from ticket class, gender, age, and fare. It combines a
scikit-learn preprocessing and classification pipeline with an interactive
web dashboard.

## Objective

Train a binary classifier using historical Titanic passenger data, evaluate it
on a held-out test set, and use the saved model to display a passenger's
predicted outcome and estimated survival probability.

## Technologies

- Python 3.10+
- Streamlit
- scikit-learn
- Pandas and NumPy
- Joblib

## Dataset

`data/titanic.csv` contains 891 passenger records from the Titanic dataset
published with the Seaborn project. The training script uses the `survived`,
`pclass`, `sex`, `age`, and `fare` columns. Missing feature values are imputed
within the model pipeline; categorical gender values are one-hot encoded.

Source: [Seaborn Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).

## Installation

From this directory, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

The script uses `data/titanic.csv`, performs a stratified 80/20 train/test
split, trains a logistic regression pipeline, prints accuracy, precision,
recall, and F1-score, and saves `model.joblib`. Optional `--data` and
`--output` arguments let you select another CSV or model destination.

### Holdout evaluation

Results below are produced by `train_model.py` using a 20% stratified test
split and `random_state=42`.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.793 |
| Precision | 0.750 |
| Recall | 0.696 |
| F1-score | 0.722 |

## Run the application

Train the model first, then launch the dashboard:

```bash
streamlit run app.py
```

Choose a passenger's ticket class, gender, age, and fare, then select
**Predict survival**. The app displays the predicted class and survival
probability. The prediction is an educational estimate based on historical
data, not a certainty.

## View all prediction outputs

Launch the separate batch-results website in another terminal:

```bash
streamlit run outputs_app.py
```

It displays the saved held-out evaluation metrics, predictions and survival
probabilities for every passenger in `data/titanic.csv`, actual outcomes,
summary charts, and a filterable results table. Use **Download all predictions
as CSV** to export the complete unfiltered batch. The table includes passengers
from both the training and test sets, so its correctness summary is in-sample;
use the held-out test-set metrics for model evaluation.
