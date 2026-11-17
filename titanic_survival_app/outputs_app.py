"""Explore batch predictions for every passenger in the Titanic dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "Fare"]
DATA_PATH = Path(__file__).resolve().parent / "data" / "titanic.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

st.set_page_config(
    page_title="Titanic Prediction Results",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4f7fb 0%, #edf3f7 100%); }
    .block-container { max-width: 1400px; padding-top: 2rem; }
    .hero {
        padding: 1.7rem 2rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg, #12304a, #176b73);
        margin-bottom: 1.3rem;
    }
    .hero h1 { margin: 0 0 .4rem 0; font-size: 2.2rem; }
    .hero p { margin: 0; color: #e3f2f1; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(model_path: str) -> dict[str, Any]:
    """Load and validate the saved model."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact has an unexpected format.")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The model feature order does not match the dashboard.")
    model = artifact.get("model")
    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise ValueError("The saved model cannot generate predictions.")
    return artifact


@st.cache_data
def load_dataset(data_path: str, modified_time: float) -> pd.DataFrame:
    """Read the dataset and preserve each source row's passenger number."""
    del modified_time
    data = pd.read_csv(data_path)
    required_columns = {"survived", "pclass", "sex", "age", "fare"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    passengers = data.loc[:, ["pclass", "sex", "age", "fare", "survived"]].copy()
    passengers.columns = FEATURE_COLUMNS + ["Actual survival"]
    passengers.insert(0, "Passenger number", data.index + 1)
    passengers["Actual survival"] = pd.to_numeric(
        passengers["Actual survival"], errors="coerce"
    )
    passengers = passengers.dropna(subset=["Actual survival"])
    passengers = passengers[passengers["Actual survival"].isin([0, 1])].copy()
    passengers["Sex"] = passengers["Sex"].astype("string").str.strip().str.lower()
    passengers["Sex"] = passengers["Sex"].replace("", pd.NA)
    for column in ("Pclass", "Age", "Fare"):
        passengers[column] = pd.to_numeric(passengers[column], errors="coerce")

    if passengers.empty:
        raise ValueError("The dataset does not contain any valid passenger records.")
    return passengers.reset_index(drop=True)


st.markdown(
    """
    <section class="hero">
      <h1>Passenger Prediction Results</h1>
      <p>Browse, filter, and download model outputs for the complete Titanic dataset.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write(
    "This dashboard applies the saved model to each passenger in the source "
    "dataset. The model scores below are from its held-out test set; table "
    "predictions cover the full dataset, including rows used during training."
)

try:
    artifact = load_artifact(str(MODEL_PATH))
except FileNotFoundError:
    st.error("The trained model is missing. Run `python train_model.py` first.")
    st.stop()
except (OSError, EOFError, KeyError, TypeError, ValueError):
    st.error("The model could not be loaded. Retrain it with `python train_model.py`.")
    st.stop()

try:
    data_modified_time = DATA_PATH.stat().st_mtime
    passengers = load_dataset(str(DATA_PATH), data_modified_time)
except FileNotFoundError:
    st.error("The Titanic dataset is missing from the project's `data` folder.")
    st.stop()
except (OSError, ValueError, pd.errors.ParserError):
    st.error("The Titanic dataset could not be read. Check `data/titanic.csv`.")
    st.stop()

try:
    model = artifact["model"]
    passenger_features = passengers.loc[:, FEATURE_COLUMNS]
    predicted_classes = model.predict(passenger_features)
    probabilities = model.predict_proba(passenger_features)
    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError("The model does not include a survival class.")
    survival_probabilities = probabilities[:, class_labels.index(1)]
    if len(predicted_classes) != len(passengers):
        raise ValueError("The model returned an incomplete set of predictions.")
except (IndexError, KeyError, TypeError, ValueError):
    st.error("Predictions could not be generated for this dataset.")
    st.stop()

results = passengers.copy()
results["Predicted survival"] = predicted_classes.astype(int)
results["Survival probability"] = survival_probabilities
results["Prediction result"] = results["Predicted survival"].map(
    {0: "Not survived", 1: "Survived"}
)
results["Actual outcome"] = results["Actual survival"].astype(int).map(
    {0: "Not survived", 1: "Survived"}
)
results["Correct prediction"] = (
    results["Actual survival"].astype(int) == results["Predicted survival"]
).map({True: "Correct", False: "Incorrect"})

metrics = artifact.get("metrics", {})
st.subheader("Held-out test-set evaluation")
metric_columns = st.columns(4)
for column, key, label in zip(
    metric_columns,
    ("accuracy", "precision", "recall", "f1_score"),
    ("Accuracy", "Precision", "Recall", "F1-score"),
):
    value = metrics.get(key)
    column.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "N/A")

st.caption(
    f"Model evaluation is from the held-out test set. Batch table below contains "
    f"{len(results):,} source passengers; its correctness summary is in-sample "
    "and is not a generalization score."
)

st.subheader("Batch prediction summary")
survivor_count = int(results["Predicted survival"].sum())
not_survivor_count = len(results) - survivor_count
correct_count = int((results["Correct prediction"] == "Correct").sum())
summary_columns = st.columns(4)
summary_columns[0].metric("Passengers", f"{len(results):,}")
summary_columns[1].metric("Predicted survivors", f"{survivor_count:,}")
summary_columns[2].metric("Predicted non-survivors", f"{not_survivor_count:,}")
summary_columns[3].metric(
    "In-sample correct", f"{correct_count / len(results):.1%}"
)

chart_column, accuracy_column = st.columns(2)
with chart_column:
    st.markdown("#### Predicted outcomes")
    chart_data = pd.DataFrame(
        {
            "Outcome": ["Survived", "Not survived"],
            "Passengers": [survivor_count, not_survivor_count],
        }
    ).set_index("Outcome")
    st.bar_chart(chart_data, color="#176b73")
with accuracy_column:
    st.markdown("#### Actual vs predicted")
    comparison = pd.crosstab(
        results["Actual outcome"], results["Prediction result"]
    ).reindex(
        index=["Not survived", "Survived"],
        columns=["Not survived", "Survived"],
        fill_value=0,
    )
    st.dataframe(comparison, width="stretch")

st.subheader("All passenger outputs")
filter_columns = st.columns(3)
with filter_columns[0]:
    selected_class = st.multiselect(
        "Ticket class",
        options=[1, 2, 3],
        default=[1, 2, 3],
        format_func=lambda value: f"Class {value}",
    )
with filter_columns[1]:
    selected_gender = st.multiselect(
        "Gender",
        options=["female", "male"],
        default=["female", "male"],
        format_func=str.title,
    )
with filter_columns[2]:
    selected_outcomes = st.multiselect(
        "Predicted outcome",
        options=["Survived", "Not survived"],
        default=["Survived", "Not survived"],
    )

filtered_results = results[
    results["Pclass"].isin(selected_class)
    & results["Sex"].isin(selected_gender)
    & results["Prediction result"].isin(selected_outcomes)
]
visible_columns = [
    "Passenger number",
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Actual outcome",
    "Prediction result",
    "Survival probability",
    "Correct prediction",
]
st.caption(f"Showing {len(filtered_results):,} of {len(results):,} passengers.")
st.dataframe(
    filtered_results.loc[:, visible_columns],
    width="stretch",
    hide_index=True,
    column_config={
        "Survival probability": st.column_config.ProgressColumn(
            "Survival probability", format="%.1%%", min_value=0, max_value=1
        )
    },
)
st.download_button(
    "Download all predictions as CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="titanic_predictions.csv",
    mime="text/csv",
    width="stretch",
)
