"""Streamlit dashboard for Titanic survival predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "Fare"]
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4f7fb 0%, #edf3f7 100%); }
    .block-container { max-width: 900px; padding-top: 2.5rem; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg, #12304a, #176b73);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: 2.25rem; }
    .hero p { margin: 0; color: #e3f2f1; font-size: 1.05rem; }
    div[data-testid="stForm"] {
        padding: 1.4rem; border: 1px solid #dce5eb;
        border-radius: 16px; background: white;
    }
    div.stButton > button {
        background: #176b73; color: white; border: 0;
        border-radius: 9px; font-weight: 600; min-height: 2.8rem;
    }
    div.stButton > button:hover { background: #12565d; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str) -> dict[str, Any]:
    """Load and validate the model artifact once per Streamlit process."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact has an unexpected format.")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The model feature order does not match the application.")
    if not hasattr(artifact.get("model"), "predict_proba"):
        raise ValueError("The saved model cannot calculate prediction probabilities.")
    return artifact


st.markdown(
    """
    <section class="hero">
      <h1>Titanic Survival Predictor</h1>
      <p>Explore how passenger details relate to survival using a machine learning model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Enter a few passenger details below to estimate the likelihood of survival. "
    "This educational model is based on historical data and is not a statement "
    "about the value of any individual."
)

try:
    artifact = load_model(str(MODEL_PATH))
except FileNotFoundError:
    st.error(
        "The trained model is not available yet. From the project folder, run "
        "`python train_model.py` and then restart the app."
    )
    st.stop()
except (OSError, EOFError, KeyError, TypeError, ValueError):
    st.error(
        "The model file could not be loaded. Please retrain it with "
        "`python train_model.py` and restart the app."
    )
    st.stop()

with st.form("passenger_details"):
    st.subheader("Passenger details")
    first_column, second_column = st.columns(2)
    with first_column:
        passenger_class = st.selectbox(
            "Ticket class",
            options=[1, 2, 3],
            format_func=lambda value: f"Class {value}",
        )
        age = st.slider("Age", min_value=0, max_value=100, value=29, step=1)
    with second_column:
        sex = st.radio("Gender", options=["Female", "Male"], horizontal=True)
        fare = st.slider(
            "Fare amount",
            min_value=0.0,
            max_value=512.33,
            value=32.20,
            step=0.5,
            format="$%.2f",
        )
    submitted = st.form_submit_button("Predict survival", use_container_width=True)

if submitted:
    values = {
        "Pclass": passenger_class,
        "Sex": sex.lower(),
        "Age": age,
        "Fare": fare,
    }
    if not all(np.isfinite(value) for value in (age, fare)):
        st.error("Please enter valid age and fare values.")
    else:
        passenger = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        model = artifact["model"]
        try:
            predicted_class = int(model.predict(passenger)[0])
            probabilities = model.predict_proba(passenger)[0]
            class_labels = list(model.classes_)
            if 1 not in class_labels:
                raise ValueError("The model does not include a survival class.")
            survival_probability = float(probabilities[class_labels.index(1)])
            if predicted_class not in (0, 1) or not 0 <= survival_probability <= 1:
                raise ValueError("The model returned an invalid prediction.")
        except (IndexError, KeyError, TypeError, ValueError):
            st.error(
                "A prediction could not be generated. Please check the passenger "
                "details and try again."
            )
        else:
            st.divider()
            st.subheader("Prediction")
            st.metric(
                "Estimated survival probability",
                f"{survival_probability:.1%}",
            )
            if predicted_class == 1:
                st.success(
                    "The model predicts that this passenger would have survived."
                )
            else:
                st.warning(
                    "The model predicts that this passenger would not have survived."
                )
            st.caption(
                "This is a statistical estimate from a historical dataset, "
                "not a certainty."
            )
