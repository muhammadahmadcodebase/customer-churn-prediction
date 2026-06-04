from __future__ import annotations

from typing import Any


def calculate_health_score(
    churn_probability: float,
    tenure: int,
    contract_type: str,
    total_services: int,
) -> int:
    base_score = 100 - int(round(churn_probability * 100))
    tenure_bonus = min(15, max(0, tenure // 4))
    service_bonus = min(15, max(0, total_services * 2))

    contract_type = str(contract_type or "").strip().lower()
    if "two year" in contract_type:
        contract_bonus = 10
    elif "one year" in contract_type:
        contract_bonus = 5
    else:
        contract_bonus = -10

    score = base_score + tenure_bonus + service_bonus + contract_bonus
    score = max(0, min(100, score))
    return score


def health_score_category(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Moderate"
    return "Poor"
