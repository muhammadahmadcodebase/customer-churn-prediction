import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

# Patch sklearn to prevent unpickling issues for newer/different sklearn versions
try:
    import collections
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        class _RemainderColsList(collections.UserList):
            def __init__(self, columns, *args, **kwargs):
                super().__init__(columns)
                self.future_dtype = kwargs.get("future_dtype", None)
                self.warning_was_emitted = kwargs.get("warning_was_emitted", False)
                self.warning_enabled = kwargs.get("warning_enabled", True)
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except (ImportError, AttributeError):
    pass


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

DEFAULT_THRESHOLD = 0.30

MODEL_CONFIGS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "paths": [MODELS_DIR / "logistic_regression.pkl"],
    },
    "lightgbm": {
        "label": "LightGBM",
        "paths": [MODELS_DIR / "lightgbm_champion.pkl", MODELS_DIR / "lightgbm_model.pkl"],
    },
    "xgboost": {
        "label": "XGBoost",
        "paths": [MODELS_DIR / "xgboost_tuned.pkl", MODELS_DIR / "xgboost_model.pkl"],
    },
    "catboost": {
        "label": "CatBoost",
        "paths": [MODELS_DIR / "catboost_tuned.pkl", MODELS_DIR / "catboost_model.pkl"],
    },
}


def _model_path(config: dict) -> Optional[Path]:
    for path in config["paths"]:
        if path.exists():
            return path
    return None


def available_model_configs() -> dict:
    available = {
        key: {**config, "path": _model_path(config)}
        for key, config in MODEL_CONFIGS.items()
        if _model_path(config) is not None
    }
    return available or {
        key: {**config, "path": config["paths"][0]} for key, config in MODEL_CONFIGS.items()
    }


def render_sidebar() -> None:
    """Render the shared project sidebar used across every Streamlit page."""
    logo_path = ASSETS_DIR / "logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=112)

    st.sidebar.markdown("## Customer Churn Prediction")
    st.sidebar.caption("Business-ready ML intelligence dashboard")

    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("app.py", label="Home", icon=":material/home:")
    st.sidebar.page_link("pages/1_Dashboard.py", label="Executive Dashboard", icon=":material/dashboard:")
    st.sidebar.page_link("pages/2_Customer_Analysis.py", label="Customer Analysis", icon=":material/person_search:")
    st.sidebar.page_link("pages/customer_prediction.py", label="Customer Prediction", icon=":material/insights:")
    st.sidebar.page_link("pages/3_Batch_Prediction.py", label="Batch Prediction", icon=":material/upload_file:")
    st.sidebar.page_link("pages/4_Model_Performance.py", label="Model Performance", icon=":material/leaderboard:")
    st.sidebar.page_link("pages/5_About_Project.py", label="About Project", icon=":material/info:")

    st.sidebar.markdown("### Model Selector")
    configs = available_model_configs()
    labels = [config["label"] for config in configs.values()]
    default_index = labels.index("XGBoost") if "XGBoost" in labels else 0
    selected_label = st.sidebar.selectbox(
        "Choose prediction model",
        labels,
        index=default_index,
        key="selected_model_label",
    )
    selected_key = next(
        key for key, config in configs.items() if config["label"] == selected_label
    )
    st.session_state["selected_model_key"] = selected_key

    st.sidebar.markdown("---")
    threshold = get_model_threshold(selected_key)
    st.sidebar.caption(f"Active churn threshold: {threshold:.2f}")


def get_selected_model_key() -> str:
    return st.session_state.get("selected_model_key", "xgboost")


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    config = available_model_configs().get(model_key)
    if not config:
        st.error(f"Unknown model key: {model_key}")
        return None

    model_path = config["path"]
    if not model_path.exists():
        st.warning(f"Model file not found: {model_path.name}")
        return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.warning(
            f"Could not load {model_path.name}. Check that the required library versions "
            f"are installed. Details: {exc}"
        )
        return None


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if not preprocessor_path.exists():
        st.warning("Preprocessor file not found. Upload the trained preprocessor.pkl artifact.")
        return None

    try:
        return joblib.load(preprocessor_path)
    except Exception as exc:
        st.warning(
            "Could not load preprocessor.pkl. This usually means the app environment does "
            f"not match the training environment. Details: {exc}"
        )
        return None


@st.cache_data(show_spinner=False)
def load_telco_data() -> pd.DataFrame:
    for file_name in ["telco_churn.csv", "dashboard_data.csv"]:
        path = DATA_DIR / file_name
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not read {file_name}: {exc}")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    path = DATA_DIR / "dashboard_data.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read dashboard_data.csv: {exc}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_feature_names() -> list[str]:
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    if not feature_names_path.exists():
        return []

    try:
        return [str(feature) for feature in joblib.load(feature_names_path)]
    except Exception as exc:
        st.warning(f"Could not load feature_names.pkl: {exc}")
        return []


@st.cache_data(show_spinner=False)
def load_threshold_config() -> dict:
    threshold_path = MODELS_DIR / "threshold.json"
    if not threshold_path.exists():
        return {}

    try:
        with threshold_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        st.warning(f"Could not read threshold.json: {exc}")
        return {}


def get_model_threshold(model_key: str) -> float:
    thresholds = load_threshold_config()
    config = MODEL_CONFIGS.get(model_key, {})
    label = str(config.get("label", ""))
    candidates = [
        model_key,
        model_key.lower(),
        label,
        label.lower(),
        label.replace(" ", "_").lower(),
        "best_threshold",
        "threshold",
        "default",
    ]

    for key in candidates:
        if key in thresholds:
            try:
                threshold = float(thresholds[key])
                if 0 <= threshold <= 1:
                    return threshold
            except (TypeError, ValueError):
                continue

    return DEFAULT_THRESHOLD


def get_training_columns(preprocessor: Optional[object]) -> list[str]:
    if preprocessor is None:
        return []

    feature_names = getattr(preprocessor, "feature_names_in_", None)
    if feature_names is None:
        return []

    return [column for column in feature_names if column not in {"Churn", "customerID"}]
