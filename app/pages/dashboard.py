
import streamlit as st
import pandas as pd

from app.services.database import get_history_collection


def get_user_history(user_id):
    """Fetch prediction history belonging to the logged-in user."""

    try:
        collection = get_history_collection()

        records = list(
            collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(100)
        )

        return records, None

    except Exception:
        return [], "Unable to load your prediction history."


def get_prediction_summary(records):
    total = len(records)

    stress_count = sum(
        1 for record in records
        if str(record.get("prediction", "")).lower() == "stress"
    )

    no_stress_count = sum(
        1 for record in records
        if str(record.get("prediction", "")).lower() == "no stress"
    )

    return total, stress_count, no_stress_count


def format_date(value):
    if value:
        return value.strftime("%d %b %Y, %I:%M %p")

    return "Unavailable"


def load_dashboard_styles():
    """Apply the same light theme as the authentication screen."""

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F1F5F9 !important;
            color: #102A43 !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }


        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] * {
            color: #102A43;
        }

        .dashboard-title {
            color: #102A43;
            font-size: 32px;
            font-weight: 750;
            letter-spacing: -1px;
            margin-bottom: 4px;
        }

        .dashboard-subtitle {
            color: #627D98;
            font-size: 14px;
            margin-bottom: 28px;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 13px;
            padding: 20px;
            min-height: 125px;
        }

        .metric-label {
            color: #627D98;
            font-size: 13px;
            font-weight: 600;
        }

        .metric-value {
            color: #102A43;
            font-size: 31px;
            font-weight: 750;
            margin-top: 8px;
        }

        .metric-note {
            color: #829AB1;
            font-size: 12px;
        }

        .section-title {
            color: #102A43;
            font-size: 19px;
            font-weight: 700;
            margin-top: 30px;
            margin-bottom: 12px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def show_dashboard():

    load_dashboard_styles()

    user = st.session_state.user
    user_id = user["user_id"]

    # Dashboard heading

    st.markdown(
        '<div class="dashboard-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="dashboard-subtitle">'
        f'Welcome back, {user["username"]}. '
        f"Here's your activity overview."
        f'</div>',
        unsafe_allow_html=True
    )

    # Fetch personal history

    records, error = get_user_history(user_id)

    if error:
        st.error(error)
        return

    total, stress_count, no_stress_count = (
        get_prediction_summary(records)
    )

    # Summary cards

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{total}</div>
                <div class="metric-note">All recorded predictions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Stress</div>
                <div class="metric-value">{stress_count}</div>
                <div class="metric-note">Predicted stress results</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">No Stress</div>
                <div class="metric-value">{no_stress_count}</div>
                <div class="metric-note">Predicted no-stress results</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Stress vs No Stress graph
    st.markdown(
        '<div class="section-title">Stress vs No Stress Distribution</div>',
        unsafe_allow_html=True
    )

    prediction_chart = pd.DataFrame(
        {"Predictions": [stress_count, no_stress_count]},
        index=["Stress", "No Stress"]
    )

    st.bar_chart(prediction_chart)

    # Recent predictions

    st.markdown(
        '<div class="section-title">Recent Predictions</div>',
        unsafe_allow_html=True
    )

    if records:

        history_rows = []

        for record in records[:10]:
            history_rows.append({
                "Date & Time": format_date(
                    record.get("created_at")
                ),
                "Prediction": record.get(
                    "prediction", "Unavailable"
                )
            })

        history_df = pd.DataFrame(history_rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date & Time": st.column_config.TextColumn(
                    "Date & Time"
                ),
                "Prediction": st.column_config.TextColumn(
                    "Prediction"
                )
            }
        )

    else:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    No predictions yet
                </div>
                <div class="metric-note">
                    Your prediction history will appear here
                    after your first HRV analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "StressSense is a research prototype, not a medical diagnostic tool."
    )
