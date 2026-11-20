
import io

import pandas as pd
import streamlit as st

from app.services.model_service import FEATURES, predict_hrv
from app.services.prediction_service import (
    save_predictions,
    get_user_predictions,
    delete_user_prediction,
)


def load_prediction_styles():
    """Use the same light theme as the authentication screen."""

    st.markdown(
        """
        <style>
        /* Main page headings */
.page-title,
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #102A43 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #102A43 !important;
}
        .stApp {
            background-color: #F1F5F9 !important;
            color: #102A43 !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }


        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] * {
            color: #102A43;
        }

        .page-title {
            color: #102A43;
            font-size: 32px;
            font-weight: 750;
            letter-spacing: -1px;
            margin-bottom: 4px;
        }

        .page-subtitle {
            color: #627D98;
            font-size: 14px;
            margin-bottom: 24px;
        }

        .result-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 13px;
            padding: 22px;
            margin-top: 14px;
        }

        .result-label {
            color: #627D98;
            font-size: 13px;
            font-weight: 600;
        }

        .result-value {
            color: #102A43;
            font-size: 27px;
            font-weight: 750;
            margin-top: 8px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def show_prediction_result(predictions):
    """Display predicted classes without probabilities or confidence scores."""

    stress_count = predictions.count("Stress")
    no_stress_count = predictions.count("No Stress")

    st.markdown("### Prediction results")

    if len(predictions) == 1:
        if predictions[0] == "Stress":
            st.error("**Predicted class: Stress**")
        else:
            st.success("**Predicted class: No Stress**")

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total records", len(predictions))
        col2.metric("Stress", stress_count)
        col3.metric("No Stress", no_stress_count)


def make_download_dataframe(dataframe, predictions):
    """Combine original HRV values and predicted labels."""

    result = dataframe.copy()
    result["Prediction"] = predictions

    return result


def manual_prediction(user_id):
    """Collect all 34 HRV features and predict using the existing model."""

    st.markdown("### Enter HRV feature values")

    st.caption(
        "Enter the 34 HRV features below. Use values calculated using "
        "the same feature-extraction methodology as the training dataset."
    )

    with st.form("manual_hrv_prediction_form"):

        values = {}

        left_column, right_column = st.columns(2)

        for index, feature in enumerate(FEATURES):
            target_column = left_column if index % 2 == 0 else right_column

            with target_column:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.8f",
                    key=f"manual_feature_{feature}"
                )

        submitted = st.form_submit_button(
            "Run manual prediction",
            use_container_width=True
        )

    if not submitted:
        return

    dataframe = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    with st.spinner("Analyzing HRV features..."):
        success, predictions = predict_hrv(dataframe)

    if not success:
        st.error(predictions)
        return

    show_prediction_result(predictions)

    saved, message = save_predictions(
        user_id,
        dataframe,
        predictions,
        source="manual"
    )

    if saved:
        st.success(message)
    else:
        st.error(message)


def csv_prediction(user_id):
    st.markdown("### Upload HRV dataset")

    st.caption(
        "Upload a CSV file containing all 34 required HRV feature columns. "
        "Each row will be processed as one prediction."
    )

    st.download_button(
        "Download CSV template",
        data=pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"),
        file_name="stresssense_hrv_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Choose an HRV CSV file",
        type=["csv"],
        key="hrv_csv_upload"
    )

    if uploaded_file is None:
        return

    try:
        dataframe = pd.read_csv(uploaded_file)

    except Exception as error:
        st.error(f"Unable to read CSV file: {error}")
        return

    missing_features = [
        feature for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        st.error(
            "Your CSV is missing these required columns: "
            + ", ".join(missing_features)
        )
        return

    if dataframe.empty:
        st.warning("The uploaded CSV contains no records.")
        return

    st.write(f"**Records detected:** {len(dataframe)}")

    st.dataframe(
        dataframe[FEATURES].head(5),
        use_container_width=True,
        hide_index=True
    )

    if len(dataframe) > 10000:
        st.error(
            "Please upload a file containing no more than 10,000 rows."
        )
        return

    if st.button(
        "Run batch prediction",
        use_container_width=True,
        key="run_csv_prediction"
    ):

        with st.spinner("Processing HRV records..."):

            success, predictions = predict_hrv(dataframe)

        if not success:
            st.error(predictions)
            return

        show_prediction_result(predictions)

        saved, message = save_predictions(
            user_id,
            dataframe[FEATURES],
            predictions,
            source="csv"
        )

        if saved:
            st.success(message)
        else:
            st.error(message)

        result_dataframe = make_download_dataframe(
            dataframe,
            predictions
        )

        csv_data = result_dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download batch prediction results",
            data=csv_data,
            file_name="stresssense_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_batch_results"
        )


def prediction_history(user_id):
    st.markdown("### Your prediction history")

    records, error = get_user_predictions(
        user_id,
        limit=100
    )

    if error:
        st.error(error)
        return

    if not records:
        st.info("You have no saved predictions yet.")
        return

    history_rows = []

    for record in records:
        history_rows.append({
            "Date and time": (
                record["created_at"].strftime("%d %b %Y, %I:%M %p")
                if record.get("created_at")
                else "Unavailable"
            ),
            "Prediction": record.get("prediction", "Unavailable"),
            "Source": record.get("source", "Unknown")
        })

    history_dataframe = pd.DataFrame(history_rows)

    st.dataframe(
        history_dataframe,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing your latest 100 saved records."
    )

    st.markdown("#### View or delete a record")

    record_options = {
        (
            f"{record.get('prediction', 'Unknown')} — "
            f"{record.get('created_at').strftime('%d %b %Y, %I:%M %p') if record.get('created_at') else 'Date unavailable'} "
            f"— {str(record['_id'])[-6:]}"
        ): str(record["_id"])
        for record in records
    }

    selected_label = st.selectbox(
        "Select a prediction record",
        options=list(record_options.keys())
    )

    selected_record_id = record_options[selected_label]

    selected_record = next(
        record for record in records
        if str(record["_id"]) == selected_record_id
    )

    with st.expander("View saved HRV input values"):
        feature_values = selected_record.get("features", {})

        feature_dataframe = pd.DataFrame(
            list(feature_values.items()),
            columns=["HRV feature", "Value"]
        )

        st.dataframe(
            feature_dataframe,
            use_container_width=True,
            hide_index=True
        )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this prediction record.",
        key=f"confirm_delete_{selected_record_id}"
    )

    if st.button(
        "Delete selected record",
        disabled=not confirm_delete,
        type="secondary"
    ):

        deleted, message = delete_user_prediction(
            user_id,
            selected_record_id
        )

        if deleted:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


def show_prediction():
    """Main StressSense prediction page."""

    load_prediction_styles()

    user = st.session_state.user
    user_id = user["user_id"]

    st.markdown(
        '<div class="page-title">HRV Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Analyze HRV features and view your predicted stress class.'
        '</div>',
        unsafe_allow_html=True
    )

    manual_tab, csv_tab, history_tab = st.tabs([
        "Manual Input",
        "CSV Upload",
        "Prediction History"
    ])

    with manual_tab:
        manual_prediction(user_id)

    with csv_tab:
        csv_prediction(user_id)

    with history_tab:
        prediction_history(user_id)

    st.caption(
        "Research prototype only. Predictions are not medical diagnoses."
    )
