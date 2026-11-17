"""Train and evaluate the Titanic survival classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "Fare"]
NUMERIC_FEATURES = ["Pclass", "Age", "Fare"]
CATEGORICAL_FEATURES = ["Sex"]
TARGET_COLUMN = "Survived"


def build_pipeline() -> Pipeline:
    """Create preprocessing and classification steps as one reusable pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(data_path: Path, model_path: Path) -> dict[str, Any]:
    """Train the model, print holdout metrics, and save a reusable artifact."""
    data = pd.read_csv(data_path)
    required_columns = {"survived", "pclass", "sex", "age", "fare"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    passengers = data.loc[:, ["pclass", "sex", "age", "fare", "survived"]].copy()
    passengers.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    passengers[TARGET_COLUMN] = pd.to_numeric(
        passengers[TARGET_COLUMN], errors="coerce"
    )
    passengers = passengers.dropna(subset=[TARGET_COLUMN])
    passengers = passengers[passengers[TARGET_COLUMN].isin([0, 1])]
    passengers["Sex"] = passengers["Sex"].astype("string").str.strip().str.lower()
    passengers["Sex"] = passengers["Sex"].replace("", pd.NA)

    if passengers.empty or passengers[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Dataset must contain passengers from both survival classes.")

    features = passengers.loc[:, FEATURE_COLUMNS]
    target = passengers[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
        },
        model_path,
    )

    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print("Holdout metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
    print(f"Saved model to: {model_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "titanic.csv",
        help="Path to the Titanic CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "model.joblib",
        help="Path where the trained model artifact will be saved.",
    )
    arguments = parser.parse_args()

    try:
        train_model(arguments.data, arguments.output)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
