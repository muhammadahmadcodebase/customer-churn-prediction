from pathlib import Path

import streamlit as st

from utils.model_loader import render_sidebar
from utils.visualizations import apply_global_styles


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

BASE_DIR = Path(__file__).resolve().parent
BANNER_PATH = BASE_DIR / "assets" / "banner.png"

if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), width='stretch')

st.markdown(
    """
    <div class="hero-panel">
        <p class="eyebrow">Machine Learning Portfolio Project</p>
        <h1>Customer Churn Prediction Dashboard</h1>
        <p>
            A production-ready Streamlit analytics app for identifying high-risk
            customers, estimating revenue exposure, and prioritizing retention actions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Open the Executive Dashboard from the sidebar to begin. Replace the placeholder "
    "pickle files in the models folder with your trained Google Colab artifacts before deployment."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Dashboard.py", label="Executive Dashboard", icon=":material/dashboard:")
with col2:
    st.page_link("pages/customer_prediction.py", label="Customer Prediction", icon=":material/insights:")
with col3:
    st.page_link("pages/2_Customer_Analysis.py", label="Customer Analysis", icon=":material/person_search:")

col4, col5 = st.columns(2)
with col4:
    st.page_link("pages/3_Batch_Prediction.py", label="Batch Prediction", icon=":material/upload_file:")
with col5:
    st.page_link("pages/4_Model_Performance.py", label="Model Performance", icon=":material/leaderboard:")
