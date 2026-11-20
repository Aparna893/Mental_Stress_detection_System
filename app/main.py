
import streamlit as st

from app.pages.authentication import show_authentication
from app.pages.dashboard import show_dashboard
from app.pages.prediction import show_prediction
from app.pages.analytics import show_analytics


def initialize_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "user" not in st.session_state:
        st.session_state.user = None

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Dashboard"


def show_navigation():
    user = st.session_state.user

    with st.sidebar:
        st.title("StressSense")
        st.caption(f"Welcome, {user['username']}")

        selected_page = st.radio(
            "Navigation",
            ["Dashboard", "Prediction", "Analytics"],
            key="active_page"
        )

        st.divider()

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.auth_mode = "Login"
            st.session_state.active_page = "Dashboard"
            st.rerun()

    return selected_page


def run_application():
    st.set_page_config(
        page_title="StressSense",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session()

    if not st.session_state.authenticated:
        show_authentication()
        return

    selected_page = show_navigation()

    if selected_page == "Dashboard":
        show_dashboard()

    elif selected_page == "Prediction":
        show_prediction()

    elif selected_page == "Analytics":
        show_analytics()


if __name__ == "__main__":
    run_application()
