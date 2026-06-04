import json
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.feature_engineering import generate_features
from utils.health_score import calculate_health_score, health_score_category
from utils.model_loader import render_sidebar
from utils.prediction_engine import (
    format_probability,
    get_retention_priority,
    load_saved_model,
    load_saved_preprocessor,
    load_shap_explainer,
    predict_customer,
    shap_top_factors,
)
from utils.recommendation_engine import build_retention_recommendations
from utils.report_generator import (
    build_csv_report,
    build_json_report,
    build_pdf_report,
)
from utils.visualizations import apply_global_styles


st.set_page_config(
    page_title="Customer Prediction",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

st.title("Customer Prediction")
st.caption("Enter a customer profile and receive a complete churn risk analysis.")

model = load_saved_model()
preprocessor = load_saved_preprocessor()

with st.form(key="customer_prediction_form"):
    st.markdown("### Customer Information")
    info_cols = st.columns(4)
    gender = info_cols[0].selectbox("Gender", ["Male", "Female"], index=0)
    senior_citizen = info_cols[1].selectbox("Senior Citizen", ["No", "Yes"], index=0)
    partner = info_cols[2].selectbox("Partner", ["No", "Yes"], index=0)
    dependents = info_cols[3].selectbox("Dependents", ["No", "Yes"], index=0)

    st.markdown("### Subscription Information")
    subscription_cols = st.columns(3)
    tenure = subscription_cols[0].number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12, step=1
    )
    phone_service = subscription_cols[1].selectbox("Phone Service", ["No", "Yes"], index=1)
    multiple_lines = subscription_cols[2].selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"],
        index=1,
    )

    st.markdown("### Service Information")
    service_cols = st.columns(4)
    internet_service = service_cols[0].selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        index=0,
    )
    online_security = service_cols[1].selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
    online_backup = service_cols[2].selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
    device_protection = service_cols[3].selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)

    support_cols = st.columns(3)
    tech_support = support_cols[0].selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
    streaming_tv = support_cols[1].selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
    streaming_movies = support_cols[2].selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)

    st.markdown("### Billing Information")
    billing_cols = st.columns(3)
    contract = billing_cols[0].selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        index=0,
    )
    paperless_billing = billing_cols[1].selectbox("Paperless Billing", ["No", "Yes"], index=1)
    payment_method = billing_cols[2].selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=0,
    )

    st.markdown("### Financial Information")
    finance_cols = st.columns(2)
    monthly_charges = finance_cols[0].number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.5,
        format="%.2f",
    )
    total_charges = finance_cols[1].number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=1.0,
        format="%.2f",
    )

    submitted = st.form_submit_button("Predict Churn")

if not submitted:
    st.info("Complete the customer profile and click Predict Churn to view instant analytics.")
    st.stop()

if monthly_charges <= 0:
    st.error("Monthly Charges must be a positive value.")
    st.stop()

if total_charges < 0:
    st.error("Total Charges cannot be negative.")
    st.stop()

input_payload = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": float(monthly_charges),
    "TotalCharges": float(total_charges),
}

try:
    prediction_df = predict_customer(input_payload)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

record = prediction_df.iloc[0].to_dict()
shap_explainer, shap_error = load_shap_explainer(model, preprocessor, pd.DataFrame([input_payload]))
shap_factors = {"positive": [], "negative": []}
if shap_explainer is not None:
    try:
        shap_factors = shap_top_factors(shap_explainer)
    except Exception:
        shap_factors = {"positive": [], "negative": []}
health_score = calculate_health_score(
    record["churn_probability"],
    int(record.get("tenure", 0)),
    record.get("Contract", "Month-to-month"),
    int(record.get("TotalServices", 0)),
)
health_category = health_score_category(health_score)
recommendations = build_retention_recommendations(record, record["churn_probability"])
priority_label = get_retention_priority(record["risk_segment"])

cards = st.container()
with cards:
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <p>Churn Probability</p>
                <h3>{format_probability(record['churn_probability'])}</h3>
                <small>{record['risk_segment']} risk segment</small>
            </div>
            <div class="kpi-card">
                <p>Predicted Status</p>
                <h3>{record['predicted_status']}</h3>
                <small>Threshold: 30%</small>
            </div>
            <div class="kpi-card">
                <p>Estimated Revenue Risk</p>
                <h3>${record['annual_revenue_at_risk']:.2f}</h3>
                <small>Next 12 months projection</small>
            </div>
            <div class="kpi-card">
                <p>Retention Priority</p>
                <h3>{priority_label}</h3>
                <small>{health_category} health</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

risk_style = record.get("risk_style", {"border": "#94a3b8", "background": "#f8fafc", "text": "#334155"})

st.markdown(
    f"""
    <div class="risk-card" style="border-color:{risk_style['border']}; background:{risk_style['background']};">
        <p>Risk Badge</p>
        <h2>{record['risk_segment']}</h2>
        <span style="background:{risk_style['text']};">{record['predicted_status']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

analytics_col1, analytics_col2 = st.columns([1.1, 0.9])
with analytics_col1:
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=record["churn_probability"] * 100,
            delta={"reference": 50, "increasing": {"color": "red"}},
            gauge={
                "axis": {"range": [0, 100], "tickformat": "%"},
                "bar": {"color": risk_style["text"]},
                "steps": [
                    {"range": [0, 25], "color": "#dcfce7"},
                    {"range": [25, 50], "color": "#fef9c3"},
                    {"range": [50, 75], "color": "#ffedd5"},
                    {"range": [75, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": "Probability Gauge"},
        )
    )
    gauge_fig.update_layout(height=360, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(gauge_fig, width='stretch')

    churn_bar = go.Figure(
        go.Bar(
            x=[record["predicted_status"]],
            y=[record["churn_probability"] * 100],
            marker_color=risk_style["text"],
            text=[format_probability(record["churn_probability"])],
            textposition="outside",
        )
    )
    churn_bar.update_layout(
        title="Churn Probability Bar",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 100], tickformat="%"),
        height=330,
    )
    st.plotly_chart(churn_bar, width="stretch")

with analytics_col2:
    risk_meter = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=record["churn_probability"] * 100,
            gauge={
                "axis": {"range": [0, 100], "tickmode": "array", "tickvals": [0, 25, 50, 75, 100]},
                "bar": {"color": risk_style["text"]},
                "steps": [
                    {"range": [0, 25], "color": "#dcfce7"},
                    {"range": [25, 50], "color": "#fef9c3"},
                    {"range": [50, 75], "color": "#ffedd5"},
                    {"range": [75, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": "Risk Meter"},
        )
    )
    risk_meter.update_layout(height=360, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(risk_meter, width='stretch')

    health_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 49], "color": "#fee2e2"},
                    {"range": [50, 69], "color": "#fde68a"},
                    {"range": [70, 89], "color": "#bef264"},
                    {"range": [90, 100], "color": "#4ade80"},
                ],
            },
            title={"text": "Customer Health Score"},
        )
    )
    health_fig.update_layout(height=360, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(health_fig, width='stretch')

shap_section = st.container()
with shap_section:
    st.subheader("SHAP Explanation")
    if shap_explainer is None:
        if shap_error:
            st.warning(
                f"SHAP could not be generated: {shap_error}\n" \
                f"This can occur if the dashboard is running with a different Python environment. " \
                f"Run the app using the active interpreter: {sys.executable} -m streamlit run app.py"
            )
        else:
            st.warning(
                "SHAP explanations are unavailable. Ensure shap and model/preprocessor artifacts are installed in the active environment. "
                "Run the app with the project virtual environment."
            )
    else:
        try:
            import shap
            import matplotlib.pyplot as plt

            shap_fig = plt.figure(figsize=(8, 5))
            shap.plots.bar(shap_explainer[0], show=False)
            st.pyplot(shap_fig)
            plt.close(shap_fig)

            waterfall_fig = plt.figure(figsize=(8, 5))
            shap.plots.waterfall(shap_explainer[0], show=False)
            st.pyplot(waterfall_fig)
            plt.close(waterfall_fig)

            positive = shap_factors["positive"]
            negative = shap_factors["negative"]
            st.markdown("**Top Positive Factors**")
            st.write(pd.DataFrame(positive, columns=["Feature", "SHAP Value"]))
            st.markdown("**Top Negative Factors**")
            st.write(pd.DataFrame(negative, columns=["Feature", "SHAP Value"]))
        except Exception as exc:
            st.error(f"Unable to render SHAP plots: {exc}")

report_section = st.container()
with report_section:
    st.subheader("Customer Risk Report")
    report_cols = st.columns(3)
    customer_info = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Total Charges": f"${total_charges:.2f}",
    }

    prediction_summary = {
        "Churn Probability": format_probability(record["churn_probability"]),
        "Predicted Status": record["predicted_status"],
        "Risk Segment": record["risk_segment"],
        "Health Score": f"{health_score} ({health_category})",
        "Revenue Risk": f"${record['annual_revenue_at_risk']:.2f}",
        "Retention Priority": priority_label,
    }

    shap_summary = {
        "positive": shap_factors["positive"],
        "negative": shap_factors["negative"],
    }

    pdf_bytes = build_pdf_report(customer_info, prediction_summary, recommendations, shap_summary)
    json_payload = build_json_report(customer_info, prediction_summary, recommendations, shap_summary)
    csv_bytes = build_csv_report(customer_info, prediction_summary, recommendations, shap_summary)

    if report_cols[0].button("Generate Customer Report"):
        st.success("Customer report generated successfully. Use the export buttons below.")

    if pdf_bytes is not None:
        report_cols[1].download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="customer_churn_report.pdf",
            mime="application/pdf",
        )
    else:
        report_cols[1].warning(
            f"PDF export unavailable. Install the PDF dependency in the active environment with "
            f"`{sys.executable} -m pip install fpdf2` and launch the app using "
            f"`{sys.executable} -m streamlit run app.py`."
        )

    report_cols[2].download_button(
        "Download JSON",
        data=json_payload,
        file_name="customer_churn_summary.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="customer_churn_summary.csv",
        mime="text/csv",
    )

    st.markdown("### Recommendations")
    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.markdown("### Generated Feature Summary")
    feature_dict = generate_features(pd.DataFrame([input_payload])).iloc[0].to_dict()
    feature_display = pd.DataFrame([(k, str(v)) for k, v in feature_dict.items()], columns=["Feature", "Value"])
    st.dataframe(feature_display, width='stretch', hide_index=True)
