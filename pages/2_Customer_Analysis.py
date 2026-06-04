import pandas as pd
import streamlit as st

from utils.model_loader import (
    get_selected_model_key,
    get_model_threshold,
    load_feature_names,
    load_model,
    load_preprocessor,
    load_telco_data,
    render_sidebar,
)
from utils.prediction import make_predictions
from utils.risk_calculator import get_risk_style
from utils.visualizations import apply_global_styles


st.set_page_config(
    page_title="Customer Analysis",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

st.title("Customer Analysis")
st.caption("Look up an individual customer and review their churn profile.")

data = load_telco_data()
if data.empty or "customerID" not in data.columns:
    st.error("Customer data is missing or does not contain a customerID column.")
    st.stop()

model_key = get_selected_model_key()
threshold = get_model_threshold(model_key)
model = load_model(model_key)
preprocessor = load_preprocessor()
feature_names = load_feature_names()

customer_ids = data["customerID"].astype(str).sort_values().tolist()
selected_customer = st.selectbox("Customer ID", customer_ids)

customer = data[data["customerID"].astype(str) == selected_customer].head(1)
if customer.empty:
    st.warning("No customer found for the selected ID.")
    st.stop()

try:
    scored = make_predictions(
        customer,
        model,
        preprocessor,
        threshold=threshold,
        allow_fallback=True,
        feature_names=feature_names,
    )
except Exception as exc:
    st.error(f"Unable to score this customer: {exc}")
    st.stop()

record = scored.iloc[0]
style = get_risk_style(record["risk_segment"])

profile_col, prediction_col = st.columns([1.3, 0.9])

with profile_col:
    st.subheader("Customer Profile")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Monthly Charges", f"${record.get('MonthlyCharges', 0):,.2f}")
    metric_cols[1].metric("Total Charges", f"${float(record.get('TotalCharges', 0) or 0):,.2f}")
    metric_cols[2].metric("Tenure", f"{int(record.get('tenure', 0))} months")
    metric_cols[3].metric("Contract Type", str(record.get("Contract", "Unknown")))

    details = {
        "Payment Method": record.get("PaymentMethod", "Unknown"),
        "Internet Service": record.get("InternetService", "Unknown"),
        "Tech Support": record.get("TechSupport", "Unknown"),
        "Online Security": record.get("OnlineSecurity", "Unknown"),
        "Paperless Billing": record.get("PaperlessBilling", "Unknown"),
        "Senior Citizen": record.get("SeniorCitizen", "Unknown"),
    }
    st.dataframe(
        pd.DataFrame([(k, str(v)) for k, v in details.items()], columns=["Attribute", "Value"]),
        width='stretch',
        hide_index=True,
    )

with prediction_col:
    st.subheader("Prediction")
    st.markdown(
        f"""
        <div class="risk-card" style="border-color:{style['border']}; background:{style['background']};">
            <p>Churn Probability</p>
            <h2 style="color:#000000;">{record['churn_probability']:.1%}</h2>
            <span style="background:{style['text']};">{record['risk_segment']} Risk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(record["recommendation"])
    st.caption(f"Prediction threshold: {threshold:.2f}")

# Debug Expander
from utils.feature_engineering import engineer_features
with st.expander("Debug Features"):
    st.markdown("### Input Columns")
    st.write(list(customer.columns))
    
    # Run feature engineering to determine the engineered columns
    customer_eng = engineer_features(customer)
    st.markdown("### Engineered Columns")
    st.write([c for c in customer_eng.columns if c not in customer.columns])
    
    st.markdown("### Final Columns Passed to Model")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        st.write(list(preprocessor.feature_names_in_))
    else:
        st.write("No preprocessor features available")
