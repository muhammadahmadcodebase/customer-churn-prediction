from __future__ import annotations

from typing import Any

RECOMMENDATION_DEFAULTS = [
    "Review the customer profile and validate current service preferences.",
    "Check loyalty incentives before making a retention offer.",
]


def build_retention_recommendations(customer: dict[str, Any], churn_probability: float) -> list[str]:
    recommendations: list[str] = []
    contract = str(customer.get("Contract", "")).strip().lower()
    tech_support = str(customer.get("TechSupport", "")).strip().lower()
    online_security = str(customer.get("OnlineSecurity", "")).strip().lower()
    monthly_charges = float(customer.get("MonthlyCharges", 0) or 0)
    total_services = int(customer.get("TotalServices", 0) or 0)

    if "month-to-month" in contract:
        recommendations.append("Recommend a long-term contract to stabilize churn risk.")
    elif "one year" in contract:
        recommendations.append("Lock in the customer with a loyalty reward on renewal.")
    elif "two year" in contract:
        recommendations.append("Highlight the savings of the current multi-year agreement.")

    if tech_support.lower() == "no":
        recommendations.append("Promote a tech support plan to reduce support-related churn.")
    if online_security.lower() == "no":
        recommendations.append("Offer a security package to strengthen perceived value.")

    if monthly_charges >= 80:
        recommendations.append("Place the customer on a discount or retention offer.")
    elif monthly_charges >= 60:
        recommendations.append("Review monthly bundle pricing for revenue optimization.")

    if total_services <= 2:
        recommendations.append("Cross-sell additional services to increase bundle value.")

    if churn_probability >= 0.75:
        recommendations.append("Classify as critical retention priority and assign an account manager.")
    elif churn_probability >= 0.50:
        recommendations.append("Recommend a mid-tier retention package and follow-up outreach.")

    if not recommendations:
        recommendations.extend(RECOMMENDATION_DEFAULTS)

    unique = []
    for item in recommendations:
        if item not in unique:
            unique.append(item)
    return unique
