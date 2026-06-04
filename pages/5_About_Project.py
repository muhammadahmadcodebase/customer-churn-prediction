import streamlit as st

from utils.model_loader import render_sidebar
from utils.visualizations import apply_global_styles


st.set_page_config(
    page_title="About Project",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

st.title("About Project")

st.markdown(
    """
    ## Business Problem

    Customer churn directly affects recurring revenue, acquisition efficiency, and long-term
    customer lifetime value. This dashboard turns churn predictions into a business workflow:
    identify risky customers, estimate revenue at risk, and recommend targeted retention actions.

    ## Customer Churn Prediction

    The application uses trained classification models to estimate each customer's probability
    of churn. A threshold of **0.30** is used by default because threshold optimization showed
    the strongest F1 performance around **0.30-0.35**.

    ## Dataset

    The project is designed around the Telco Customer Churn dataset, including demographic,
    account, service usage, contract, billing, tenure, monthly charge, and total charge features.

    ## Feature Engineering

    The preprocessing pipeline handles missing values, encodes categorical variables, scales
    numeric features where appropriate, and keeps inference-time transformations consistent
    with model training.

    ## Models Used

    - Logistic Regression
    - LightGBM
    - XGBoost
    - CatBoost

    ## Hyperparameter Tuning

    The trained models are intended to come from a Google Colab experimentation notebook with
    cross-validation, threshold optimization, and metric comparison across accuracy, precision,
    recall, F1, ROC-AUC, and log loss.

    ## Business Impact

    - Revenue protection through early churn detection
    - Customer retention prioritization by risk severity
    - Risk segmentation for operational customer success workflows
    - Downloadable recommendations for CRM or campaign activation

    

    
    """
)
