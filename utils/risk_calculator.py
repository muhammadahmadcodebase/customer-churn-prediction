import numpy as np
import pandas as pd


RISK_COLORS = {
    "Low": {"text": "#15803d", "background": "#dcfce7", "border": "#22c55e"},
    "Medium": {"text": "#a16207", "background": "#fef9c3", "border": "#eab308"},
    "High": {"text": "#c2410c", "background": "#ffedd5", "border": "#f97316"},
    "Critical": {"text": "#b91c1c", "background": "#fee2e2", "border": "#ef4444"},
}


def assign_risk_segment(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Critical"


def recommendation_for_segment(segment: str) -> str:
    recommendations = {
        "Low": "Customer appears stable.",
        "Medium": "Send retention email.",
        "High": "Offer discount package.",
        "Critical": "Immediate retention intervention required.",
    }
    return recommendations.get(segment, "Review customer manually.")


def get_risk_style(segment: str) -> dict[str, str]:
    return RISK_COLORS.get(segment, RISK_COLORS["Low"])


def calculate_fallback_probability(data: pd.DataFrame) -> np.ndarray:
    """Business-rule fallback used only for demo mode when artifacts are unavailable."""
    tenure = pd.to_numeric(data.get("tenure", 0), errors="coerce").fillna(0)
    monthly = pd.to_numeric(data.get("MonthlyCharges", 0), errors="coerce").fillna(0)
    contract = data.get("Contract", pd.Series(["Unknown"] * len(data))).astype(str)
    tech_support = data.get("TechSupport", pd.Series(["Unknown"] * len(data))).astype(str)
    online_security = data.get("OnlineSecurity", pd.Series(["Unknown"] * len(data))).astype(str)

    tenure_risk = np.clip((24 - tenure) / 24, 0, 1) * 0.25
    charge_risk = np.clip((monthly - 45) / 75, 0, 1) * 0.25
    contract_risk = contract.str.contains("Month-to-month", case=False, na=False).astype(float) * 0.22
    support_risk = tech_support.str.contains("No", case=False, na=False).astype(float) * 0.12
    security_risk = online_security.str.contains("No", case=False, na=False).astype(float) * 0.10

    probability = 0.08 + tenure_risk + charge_risk + contract_risk + support_risk + security_risk
    return np.clip(probability.to_numpy(), 0.02, 0.95)
