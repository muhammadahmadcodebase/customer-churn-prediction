from __future__ import annotations

import numpy as np
import pandas as pd

from utils.risk_calculator import assign_risk_segment, calculate_fallback_probability, recommendation_for_segment
from utils.feature_engineering import engineer_features


TARGET_COLUMNS = {
    "Churn",
    "churn",
    "Churn_Yes",
    "PredictedChurn",
    "predicted_churn",
    "ChurnProbability",
    "churn_probability",
    "RiskSegment",
    "risk_segment",
}
ID_COLUMNS = {"customerID", "Customer ID", "customer_id"}
DEFAULT_THRESHOLD = 0.30


def clean_input_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Telco data quality issues before inference."""
    cleaned = data.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    for column in cleaned.select_dtypes(include=["number"]).columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in cleaned.select_dtypes(include=["object", "category"]).columns:
        cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def normalize_prediction_columns(data: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Convert saved dashboard prediction columns to the app's canonical names."""
    normalized = clean_input_data(data)
    rename_map = {
        "ChurnProbability": "churn_probability",
        "PredictedChurn": "predicted_churn",
        "RiskSegment": "risk_segment",
        "Customer ID": "customerID",
    }
    normalized = normalized.rename(columns={k: v for k, v in rename_map.items() if k in normalized.columns})

    if "churn_probability" in normalized.columns:
        normalized["churn_probability"] = pd.to_numeric(
            normalized["churn_probability"], errors="coerce"
        ).fillna(0).clip(0, 1)
    if "predicted_churn" not in normalized.columns and "churn_probability" in normalized.columns:
        normalized["predicted_churn"] = (normalized["churn_probability"] >= threshold).astype(int)
    if "risk_segment" not in normalized.columns and "churn_probability" in normalized.columns:
        normalized["risk_segment"] = normalized["churn_probability"].apply(assign_risk_segment)
    if "recommendation" not in normalized.columns:
        if "Recommendation" in normalized.columns:
            normalized["recommendation"] = normalized["Recommendation"]
        elif "risk_segment" in normalized.columns:
            normalized["recommendation"] = normalized["risk_segment"].apply(recommendation_for_segment)

    if "MonthlyCharges" in normalized.columns and "churn_probability" in normalized.columns:
        monthly_charges = pd.to_numeric(normalized["MonthlyCharges"], errors="coerce").fillna(0)
        normalized["annual_revenue_at_risk"] = monthly_charges * 12 * normalized["churn_probability"]

    return normalized


def _feature_frame(data: pd.DataFrame, preprocessor: object | None) -> pd.DataFrame:
    cleaned = clean_input_data(data)

    feature_names = getattr(preprocessor, "feature_names_in_", None)
    if feature_names is not None:
        missing = [column for column in feature_names if column not in cleaned.columns]
        if missing:
            raise ValueError("Missing columns required by preprocessor: " + ", ".join(sorted(missing)))
        return cleaned.reindex(columns=feature_names, fill_value=0)

    drop_columns = [column for column in cleaned.columns if column in TARGET_COLUMNS or column in ID_COLUMNS]
    return cleaned.drop(columns=drop_columns, errors="ignore")


def _predict_probabilities(model: object, features) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Selected model does not support predict_proba().")

    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.ravel()


def make_predictions(
    data: pd.DataFrame,
    model: object | None,
    preprocessor: object | None,
    threshold: float = DEFAULT_THRESHOLD,
    allow_fallback: bool = False,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Generate churn probability, class, risk segment, and recommendation columns."""
    if data.empty:
        raise ValueError("Input data is empty.")

    cleaned = clean_input_data(data)
    cleaned = engineer_features(cleaned)

    normalized = normalize_prediction_columns(cleaned, threshold)
    if "churn_probability" in normalized.columns and (model is None or preprocessor is None):
        return normalized

    if model is None or preprocessor is None:
        if not allow_fallback:
            raise ValueError("Model or preprocessor is missing.")
        probabilities = calculate_fallback_probability(cleaned)
    else:
        feature_data = _feature_frame(cleaned, preprocessor)
        transformed = preprocessor.transform(feature_data) if hasattr(preprocessor, "transform") else feature_data
        if feature_names:
            transformed = pd.DataFrame(transformed, columns=feature_names, index=cleaned.index)
        probabilities = _predict_probabilities(model, transformed)

    probabilities = np.clip(probabilities.astype(float), 0, 1)
    result = cleaned.copy()
    result["churn_probability"] = probabilities
    result["predicted_churn"] = (result["churn_probability"] >= threshold).astype(int)
    result["risk_segment"] = result["churn_probability"].apply(assign_risk_segment)
    result["recommendation"] = result["risk_segment"].apply(recommendation_for_segment)
    if "MonthlyCharges" in result.columns:
        monthly_charges = pd.to_numeric(result["MonthlyCharges"], errors="coerce").fillna(0)
    else:
        monthly_charges = pd.Series(np.zeros(len(result)), index=result.index)
    result["annual_revenue_at_risk"] = monthly_charges * 12 * result["churn_probability"]
    return result
