
import streamlit as st

st.set_page_config(
    page_title="Sidebar Test",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sidebar Test")
st.write("If you can see this, the app is running.")

st.sidebar.title("Test Sidebar")
st.sidebar.write("Can you see this sidebar?")
st.sidebar.button("Test Button")
