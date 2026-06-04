import pandas as pd
import streamlit as st

from utils.model_loader import (
    get_selected_model_key,
    get_model_threshold,
    load_dashboard_data,
    load_feature_names,
    load_model,
    load_preprocessor,
    load_telco_data,
    render_sidebar,
)
from utils.prediction import make_predictions, normalize_prediction_columns
from utils.visualizations import (
    apply_global_styles,
    churn_distribution_pie,
    kpi_cards,
    monthly_charges_scatter,
    revenue_at_risk_chart,
    risk_segment_bar,
    tenure_scatter,
)


st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

st.title("Customer Churn Intelligence Dashboard")
st.caption("Executive view of churn exposure, customer risk, and retention priorities.")

model_key = get_selected_model_key()
threshold = get_model_threshold(model_key)
dashboard_data = load_dashboard_data()
data = dashboard_data if not dashboard_data.empty else load_telco_data()

if data.empty:
    st.error("No customer data found. Add data/telco_churn.csv or data/dashboard_data.csv.")
    st.stop()

try:
    if "ChurnProbability" in data.columns or "churn_probability" in data.columns:
        scored = normalize_prediction_columns(data, threshold=threshold)
    else:
        model = load_model(model_key)
        preprocessor = load_preprocessor()
        feature_names = load_feature_names()
        scored = make_predictions(
            data,
            model,
            preprocessor,
            threshold=threshold,
            allow_fallback=True,
            feature_names=feature_names,
        )
except Exception as exc:
    st.error(f"Unable to generate dashboard predictions: {exc}")
    st.stop()

st.caption(f"Predicted churn uses the active threshold from threshold.json: {threshold:.2f}")

kpi_cards(scored)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(churn_distribution_pie(scored), width='stretch')
with chart_col2:
    st.plotly_chart(risk_segment_bar(scored), width='stretch')

st.plotly_chart(revenue_at_risk_chart(scored), width='stretch')

scatter_col1, scatter_col2 = st.columns(2)
with scatter_col1:
    st.plotly_chart(monthly_charges_scatter(scored), width='stretch')
with scatter_col2:
    st.plotly_chart(tenure_scatter(scored), width='stretch')

st.subheader("Top 20 High-Risk Customers")
search = st.text_input("Search customer ID", placeholder="Enter customer ID")

table_columns = [
    "Customer ID",
    "Churn Probability",
    "Monthly Charges",
    "Tenure",
    "Risk Segment",
]

table = scored.copy()
table["Customer ID"] = table["customerID"].astype(str)
table["Churn Probability"] = table["churn_probability"].map(lambda value: f"{value:.1%}")
monthly_series = table["MonthlyCharges"] if "MonthlyCharges" in table.columns else pd.Series(0, index=table.index)
tenure_series = table["tenure"] if "tenure" in table.columns else pd.Series(0, index=table.index)
table["Monthly Charges"] = monthly_series.map(lambda value: f"${float(value):,.2f}")
table["Tenure"] = tenure_series.astype(int)
table["Risk Segment"] = table["risk_segment"]
table = table.sort_values("churn_probability", ascending=False).head(20)

if search:
    table = table[table["Customer ID"].str.contains(search, case=False, na=False)]

st.dataframe(table[table_columns], width='stretch', hide_index=True)
