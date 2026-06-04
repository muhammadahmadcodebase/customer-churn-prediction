import pandas as pd
import plotly.express as px
import streamlit as st

from utils.risk_calculator import RISK_COLORS


PLOTLY_TEMPLATE = "plotly_white"
RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLOR_MAP = {segment: values["text"] for segment, values in RISK_COLORS.items()}


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .hero-panel {
            padding: 1.75rem 0 1rem 0;
        }
        .hero-panel .eyebrow {
            color: #2563eb;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .hero-panel h1 {
            font-size: 2.6rem;
            line-height: 1.1;
            margin-bottom: 0.6rem;
        }
        .hero-panel p {
            max-width: 860px;
            color: #475569;
            font-size: 1.05rem;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1.25rem 0 1.5rem 0;
        }
        .kpi-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1.2rem;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .kpi-card p {
            margin: 0 0 0.35rem 0;
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .kpi-card h3 {
            margin: 0;
            color: #0f172a;
            font-size: 1.85rem;
        }
        .risk-card {
            border: 2px solid;
            border-radius: 8px;
            padding: 1.5rem;
            min-height: 220px;
        }
        .risk-card p {
            color: #475569;
            font-weight: 700;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            font-size: 0.8rem;
        }
        .risk-card h2 {
            font-size: 3rem;
            margin: 0.3rem 0 1.1rem 0;
            color: #000000 !important;
        }
        .risk-card span {
            color: white;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-weight: 700;
        }
        .best-badge {
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            background: #eff6ff;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            display: flex;
            gap: 1rem;
            align-items: baseline;
            flex-wrap: wrap;
        }
        .best-badge span {
            color: #1d4ed8;
            font-weight: 800;
            text-transform: uppercase;
            font-size: 0.78rem;
        }
        .best-badge strong {
            color: #0f172a;
            font-size: 1.4rem;
        }
        .best-badge small {
            color: #475569;
        }
        @media (max-width: 900px) {
            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 560px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            .hero-panel h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def available_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def kpi_cards(data: pd.DataFrame) -> None:
    total_customers = len(data)
    predicted_churn = int(data["predicted_churn"].sum())
    avg_probability = data["churn_probability"].mean()
    revenue_at_risk = data["annual_revenue_at_risk"].sum()

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card"><p>Total Customers</p><h3>{total_customers:,}</h3></div>
            <div class="kpi-card"><p>Predicted Churn Customers</p><h3>{predicted_churn:,}</h3></div>
            <div class="kpi-card"><p>Average Churn Probability</p><h3>{avg_probability:.1%}</h3></div>
            <div class="kpi-card"><p>Revenue At Risk</p><h3>${revenue_at_risk:,.0f}</h3></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def churn_distribution_pie(data: pd.DataFrame):
    chart_data = data.assign(Prediction=data["predicted_churn"].map({1: "Churn", 0: "Retain"}))
    fig = px.pie(
        chart_data,
        names="Prediction",
        title="Churn Distribution",
        hole=0.42,
        color="Prediction",
        color_discrete_map={"Churn": "#dc2626", "Retain": "#16a34a"},
        template=PLOTLY_TEMPLATE,
    )
    return fig


def risk_segment_bar(data: pd.DataFrame):
    chart_data = data["risk_segment"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    chart_data.columns = ["Risk Segment", "Customers"]
    fig = px.bar(
        chart_data,
        x="Risk Segment",
        y="Customers",
        title="Risk Segment Distribution",
        color="Risk Segment",
        color_discrete_map=RISK_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return fig


def revenue_at_risk_chart(data: pd.DataFrame):
    chart_data = (
        data.assign(Prediction=data["predicted_churn"].map({1: "Predicted Churn", 0: "Retain"}))
        .groupby(["risk_segment", "Prediction"], as_index=False)["annual_revenue_at_risk"]
        .sum()
    )
    fig = px.bar(
        chart_data,
        x="risk_segment",
        y="annual_revenue_at_risk",
        color="Prediction",
        category_orders={"risk_segment": RISK_ORDER},
        title="Revenue At Risk by Segment",
        labels={"risk_segment": "Risk Segment", "annual_revenue_at_risk": "Annual Revenue At Risk"},
        color_discrete_map={"Predicted Churn": "#dc2626", "Retain": "#2563eb"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(barmode="stack")
    return fig


def monthly_charges_scatter(data: pd.DataFrame):
    fig = px.scatter(
        data,
        x="MonthlyCharges",
        y="churn_probability",
        color="risk_segment",
        color_discrete_map=RISK_COLOR_MAP,
        category_orders={"risk_segment": RISK_ORDER},
        hover_data=available_columns(data, ["customerID", "tenure", "Contract"]),
        title="Monthly Charges vs Churn Probability",
        labels={"MonthlyCharges": "Monthly Charges", "churn_probability": "Churn Probability"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def tenure_scatter(data: pd.DataFrame):
    fig = px.scatter(
        data,
        x="tenure",
        y="churn_probability",
        color="risk_segment",
        color_discrete_map=RISK_COLOR_MAP,
        category_orders={"risk_segment": RISK_ORDER},
        hover_data=available_columns(data, ["customerID", "MonthlyCharges", "Contract"]),
        title="Tenure vs Churn Probability",
        labels={"tenure": "Tenure", "churn_probability": "Churn Probability"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig
