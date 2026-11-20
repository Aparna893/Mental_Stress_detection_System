from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from app.services.database import get_history_collection
from app.services.model_service import FEATURES, load_model, load_scaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "participant_hrv_34_features.csv"


@st.cache_resource
def load_personal_explainer():
    """Build an explainer using the existing model, scaler and background data."""

    model = load_model()
    scaler = load_scaler()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Background dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)

    missing = [feature for feature in FEATURES if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Background dataset is missing features: {missing}")

    background_data = dataset[FEATURES].apply(
        pd.to_numeric, errors="raise"
    )

    background_data = background_data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if background_data.empty:
        raise ValueError("No valid background samples are available.")

    background = background_data.sample(
        n=min(40, len(background_data)),
        random_state=42
    ).to_numpy(dtype=np.float32)

    def predict_probability(data):
        values = np.asarray(data, dtype=np.float32)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        dataframe = pd.DataFrame(values, columns=FEATURES)
        scaled = scaler.transform(dataframe)

        return model.predict(
            scaled, verbose=0
        ).reshape(-1)

    masker = shap.maskers.Independent(
        background,
        max_samples=len(background)
    )

    explainer = shap.Explainer(
        predict_probability,
        masker,
        algorithm="permutation",
        feature_names=FEATURES
    )

    return explainer, predict_probability


def load_user_history(user_id):
    """Read only the authenticated user's saved predictions."""

    collection = get_history_collection()

    return list(
        collection.find(
            {"user_id": str(user_id)},
            {
                "_id": 1,
                "created_at": 1,
                "prediction": 1,
                "features": 1,
                "source": 1,
            }
        ).sort("created_at", -1)
    )


def format_record(record, index):
    timestamp = record.get("created_at")

    if timestamp:
        date_text = timestamp.strftime("%d %b %Y, %I:%M %p")
    else:
        date_text = "Date unavailable"

    prediction = record.get("prediction", "Unknown")
    source = record.get("source", "Unknown")

    return f"{index + 1}. {prediction} | {date_text} | {source}"


def show_individual_explanation(record):
    """Explain one selected saved HRV prediction."""

    features = record.get("features")

    if not isinstance(features, dict):
        st.error("This prediction does not contain saved HRV feature values.")
        return

    missing = [feature for feature in FEATURES if feature not in features]

    if missing:
        st.error(
            "This record is missing required HRV features: "
            + ", ".join(missing)
        )
        return

    try:
        feature_values = pd.DataFrame(
            [[features[name] for name in FEATURES]],
            columns=FEATURES
        ).apply(pd.to_numeric, errors="raise")

        values = feature_values.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            st.error("The saved HRV values contain invalid numbers.")
            return

    except (TypeError, ValueError):
        st.error("The saved HRV feature values are not all numeric.")
        return

    st.subheader("Selected Prediction")

    saved_label = str(record.get("prediction", "Unknown"))
    st.metric("Previously saved prediction", saved_label)

    with st.spinner("Calculating this prediction's SHAP explanation..."):
        try:
            explainer, predict_probability = load_personal_explainer()

            probability = float(predict_probability(values)[0])
            current_label = "Stress" if probability >= 0.5 else "No Stress"

            explanation = explainer(
                values,
                max_evals=2 * len(FEATURES) + 1,
                batch_size=16
            )

        except Exception as error:
            st.error(f"Unable to calculate the individual explanation: {error}")
            return

    if not np.isfinite(probability):
        st.error("The model returned an invalid prediction value.")
        return

    if current_label.lower() != saved_label.strip().lower():
        st.warning(
            "The current model prediction differs from the saved label. "
            "The explanation below describes the current model and scaler "
            "output, not necessarily the model version used when the record "
            "was originally saved."
        )

    result_columns = st.columns(2)
    result_columns[0].metric("Current model prediction", current_label)
    result_columns[1].metric("Model output", f"{probability:.4f}")

    st.caption(
        "The model output is a sigmoid score, not a clinically calibrated "
        "probability or a measure of diagnostic certainty."
    )

    shap_values = np.asarray(explanation.values).reshape(-1)
    base_values = np.asarray(explanation.base_values).reshape(-1)
    base_value = float(base_values[0])

    if len(shap_values) != len(FEATURES):
        st.error("Unexpected SHAP output dimensions.")
        return

    if not np.isfinite(shap_values).all() or not np.isfinite(base_value):
        st.error("The SHAP explanation contains invalid numerical values.")
        return

    contribution_data = pd.DataFrame({
        "Feature": FEATURES,
        "HRV value": values.reshape(-1),
        "SHAP contribution": shap_values
    })

    contribution_data["Direction"] = np.where(
        contribution_data["SHAP contribution"] >= 0,
        "Pushes toward Stress",
        "Pushes toward No Stress"
    )

    contribution_data["Absolute contribution"] = (
        contribution_data["SHAP contribution"].abs()
    )

    contribution_data = contribution_data.sort_values(
        "Absolute contribution", ascending=False
    )

    st.subheader("Why did the model make this prediction?")

    st.markdown(
        f"**Base model output:** `{base_value:.5f}`  \n"
        f"**Current model output:** `{probability:.5f}`"
    )

    st.caption(
        "Positive SHAP values push the model output toward Stress. "
        "Negative SHAP values push it toward No Stress."
    )

    top_features = contribution_data.head(10).set_index("Feature")

    st.markdown("**Top 10 contributing HRV features**")
    st.bar_chart(top_features["SHAP contribution"])

    st.markdown("**Individual feature contributions**")
    st.dataframe(
        contribution_data[
            [
                "Feature",
                "HRV value",
                "SHAP contribution",
                "Direction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "SHAP values explain how the model's output changes relative to "
        "its background reference. They do not establish physiological "
        "causes of stress."
    )

    with st.expander("View individual SHAP waterfall explanation"):
        try:
            figure = shap.plots.waterfall(
                explanation[0],
                max_display=15,
                show=False
            )
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)

        except Exception as error:
            plt.close("all")
            st.warning(f"Could not render the waterfall plot: {error}")


def show_analytics():
    st.title("Personalized Analytics")
    st.caption(
        "Explore your prediction history and understand individual "
        "model predictions."
    )

    user = st.session_state.get("user") or {}
    user_id = user.get("user_id")

    if not user_id:
        st.error("Your user session could not be identified. Please log in again.")
        return

    try:
        records = load_user_history(user_id)
    except Exception as error:
        st.error(f"Unable to load your prediction history: {error}")
        return

    if not records:
        st.info(
            "You have no saved predictions yet. Make a prediction on the "
            "Prediction page to see your personalized analytics."
        )
        return

    labels = [format_record(record, i) for i, record in enumerate(records)]

    st.subheader("Select a prediction to explain")

    selected_index = st.selectbox(
        "Your saved prediction history",
        options=list(range(len(records))),
        format_func=lambda index: labels[index],
        key="personal_shap_record"
    )

    selected_record = records[selected_index]

    st.divider()
    show_individual_explanation(selected_record)


