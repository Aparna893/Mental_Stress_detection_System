
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "federated_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "federated_scaler.pkl"


# Exact feature order used during model training
FEATURES = [
    "MEAN_RR",
    "MEDIAN_RR",
    "SDRR",
    "RMSSD",
    "SDSD",
    "SDRR_RMSSD",
    "HR",
    "pNN25",
    "pNN50",
    "SD1",
    "SD2",
    "KURT",
    "SKEW",
    "MEAN_REL_RR",
    "MEDIAN_REL_RR",
    "SDRR_REL_RR",
    "RMSSD_REL_RR",
    "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR",
    "KURT_REL_RR",
    "SKEW_REL_RR",
    "VLF",
    "VLF_PCT",
    "LF",
    "LF_PCT",
    "LF_NU",
    "HF",
    "HF_PCT",
    "HF_NU",
    "TP",
    "LF_HF",
    "HF_LF",
    "sampen",
    "higuci"
]


@st.cache_resource
def load_model():
    """Load the federated-trained model once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.input_shape[-1] != len(FEATURES):
        raise ValueError(
            "Model input feature count does not match "
            "the expected 34 HRV features."
        )

    return model


@st.cache_resource
def load_scaler():
    """Load the scaler fitted during model training."""

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError(
            "Scaler feature count does not match "
            "the expected 34 HRV features."
        )

    return scaler


def validate_hrv_data(dataframe):
    """Validate the supplied HRV data."""

    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input must be a pandas DataFrame."

    missing_features = [
        feature for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        return (
            False,
            "Missing HRV features: "
            + ", ".join(missing_features)
        )

    if dataframe.empty:
        return False, "No HRV records were provided."

    try:
        feature_data = dataframe[FEATURES].apply(
            pd.to_numeric,
            errors="raise"
        )

        values = feature_data.to_numpy(dtype=np.float64)

    except (ValueError, TypeError):
        return False, "All 34 HRV features must contain numeric values."

    if not np.isfinite(values).all():
        return (
            False,
            "HRV values cannot contain missing or infinite numbers."
        )

    return True, feature_data


def predict_hrv(dataframe):
    """
    Generate Stress / No Stress predictions.

    Returns:
        success: Boolean
        result: List of predicted labels or an error message
    """

    valid, validation_result = validate_hrv_data(dataframe)

    if not valid:
        return False, validation_result

    try:
        model = load_model()
        scaler = load_scaler()

        feature_data = validation_result

        # Apply the same scaler used during training
        scaled_data = scaler.transform(feature_data)

        # Generate sigmoid predictions
        model_output = model.predict(
            scaled_data,
            verbose=0
        ).reshape(-1)

        if not np.isfinite(model_output).all():
            return False, "The model returned invalid prediction values."

        if ((model_output < 0) | (model_output > 1)).any():
            return False, "The model returned values outside the expected range."

        # Threshold: 0 = No Stress, 1 = Stress
        predicted_classes = (
            model_output >= 0.5
        ).astype(int)

        labels = [
            "Stress" if value == 1 else "No Stress"
            for value in predicted_classes
        ]

        return True, labels

    except Exception as error:
        return False, f"Prediction failed: {error}"
