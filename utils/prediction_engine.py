from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.feature_engineering import generate_features
from utils.risk_calculator import assign_risk_segment, calculate_fallback_probability
from utils.risk_calculator import get_risk_style as _get_risk_style

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_FILE = MODELS_DIR / "saved_xgboost_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "saved_preprocessor.pkl"
FALLBACK_MODEL_FILE = MODELS_DIR / "xgboost_tuned.pkl"
FALLBACK_PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"


@st.cache_resource(show_spinner=False)
def load_saved_model() -> Any | None:
    """Load the production XGBoost model from the models directory."""
    model_path = MODEL_FILE if MODEL_FILE.exists() else FALLBACK_MODEL_FILE
    if not model_path.exists():
        st.warning(
            f"Churn model not found. Looking for {MODEL_FILE.name} or fallback {FALLBACK_MODEL_FILE.name}."
        )
        return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.warning(f"Could not load model {model_path.name}. Details: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_saved_preprocessor() -> Any | None:
    """Load the inference preprocessor used at training time."""
    preprocessor_path = PREPROCESSOR_FILE if PREPROCESSOR_FILE.exists() else FALLBACK_PREPROCESSOR_FILE
    if not preprocessor_path.exists():
        st.warning(
            f"Preprocessor not found. Looking for {PREPROCESSOR_FILE.name} or fallback {FALLBACK_PREPROCESSOR_FILE.name}."
        )
        return None

    try:
        return joblib.load(preprocessor_path)
    except Exception as exc:
        st.warning(f"Could not load preprocessor {preprocessor_path.name}. Details: {exc}")
        return None


def _prepare_features(raw_data: pd.DataFrame, preprocessor: Any | None = None) -> pd.DataFrame:
    data = raw_data.copy()
    data = generate_features(data)
    if preprocessor is None:
        return data

    feature_names = getattr(preprocessor, "feature_names_in_", None)
    if feature_names is None:
        return data

    missing_columns = [column for column in feature_names if column not in data.columns]
    if missing_columns:
        raise ValueError("Missing required columns for inference: " + ", ".join(sorted(missing_columns)))

    return data.reindex(columns=feature_names, fill_value=0)


def _predict_probabilities(model: Any, features: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support probability predictions.")

    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.ravel()


def predict_customer(raw_inputs: dict[str, Any], threshold: float = 0.30) -> pd.DataFrame:
    """Generate a churn prediction for a single customer record."""
    raw_df = pd.DataFrame([raw_inputs])
    model = load_saved_model()
    preprocessor = load_saved_preprocessor()

    if raw_df.empty:
        raise ValueError("No input values were provided for prediction.")

    cleaned = raw_df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce").fillna(0.0)
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce").fillna(0.0)
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce").fillna(0).astype(int)

    engineered = generate_features(cleaned)
    if model is None or preprocessor is None:
        probabilities = calculate_fallback_probability(engineered)
    else:
        feature_data = _prepare_features(engineered, preprocessor)
        transformed = preprocessor.transform(feature_data) if hasattr(preprocessor, "transform") else feature_data
        probabilities = _predict_probabilities(model, transformed)

    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    result = engineered.copy()
    result["churn_probability"] = probabilities
    result["predicted_churn"] = (probabilities >= threshold).astype(int)
    result["risk_segment"] = result["churn_probability"].apply(assign_risk_segment)
    result["predicted_status"] = result["predicted_churn"].apply(
        lambda value: "Likely to Churn" if value == 1 else "Likely to Stay"
    )
    result["annual_revenue_at_risk"] = result["MonthlyCharges"] * 12 * result["churn_probability"]
    result["risk_style"] = result["risk_segment"].apply(_get_risk_style)
    return result


def format_probability(probability: float) -> str:
    return f"{probability * 100:.2f}%"


def get_retention_priority(segment: str) -> str:
    priorities = {
        "Low": "Monitor cadence",
        "Medium": "Proactive outreach",
        "High": "Retention campaign",
        "Critical": "Immediate action",
    }
    return priorities.get(segment, "Standard review")


def load_shap_explainer(model: Any, preprocessor: Any, raw_data: pd.DataFrame) -> tuple[Any | None, str | None]:
    try:
        import shap
    except ImportError as exc:
        return None, f"shap library is missing: {exc}"

    if model is None or preprocessor is None or raw_data.empty:
        return None, "Model or preprocessor not available for SHAP explanation."

    try:
        feature_frame = _prepare_features(raw_data, preprocessor)
        transformed = preprocessor.transform(feature_frame) if hasattr(preprocessor, "transform") else feature_frame

        feature_names = None
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out(feature_frame.columns)
            except Exception as exc:
                feature_names = None
                feature_names_error = str(exc)
        elif hasattr(preprocessor, "feature_names_in_"):
            feature_names = getattr(preprocessor, "feature_names_in_", None)

        if feature_names is not None and hasattr(transformed, "shape"):
            try:
                if getattr(transformed, "shape", None) and transformed.shape[1] == len(feature_names):
                    transformed = pd.DataFrame(transformed, columns=list(feature_names))
            except Exception:
                pass

        if hasattr(transformed, "toarray"):
            try:
                transformed = transformed.toarray()
            except Exception:
                pass

        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed)

        if getattr(explanation, "feature_names", None) is None and feature_names is not None:
            try:
                explanation.feature_names = list(feature_names)
            except Exception:
                pass

        return explanation, None
    except Exception as exc:
        return None, str(exc)


def shap_top_factors(explanation: Any, count: int = 5) -> dict[str, list[tuple[str, float]]]:
    if explanation is None:
        return {"positive": [], "negative": []}

    values = np.asarray(explanation.values[0]).ravel()
    feature_names = getattr(explanation, "feature_names", None)
    if feature_names is None:
        feature_names = [f"Feature {idx + 1}" for idx in range(values.shape[0])]
    feature_names = list(feature_names)

    shap_pairs = list(zip(feature_names, values))
    sorted_pairs = sorted(shap_pairs, key=lambda item: item[1], reverse=True)
    positive = [(name, float(value)) for name, value in sorted_pairs[:count] if value > 0]
    negative = [(name, float(value)) for name, value in sorted_pairs[-count:] if value < 0]
    return {"positive": positive, "negative": negative}


def format_engineering_summary(record: pd.Series) -> dict[str, Any]:
    return {
        "ContractRiskScore": int(record.get("ContractRiskScore", 0)),
        "TenureGroup": str(record.get("TenureGroup", "Unknown")),
        "MonthlySpendCategory": str(record.get("MonthlySpendCategory", "Unknown")),
        "RevenueSegment": str(record.get("RevenueSegment", "Unknown")),
        "HighRiskCustomer": int(record.get("HighRiskCustomer", 0)),
        "TotalServices": int(record.get("TotalServices", 0)),
    }
