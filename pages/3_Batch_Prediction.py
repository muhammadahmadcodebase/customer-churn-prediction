import pandas as pd
import streamlit as st

from utils.model_loader import (
    get_selected_model_key,
    get_model_threshold,
    get_training_columns,
    load_feature_names,
    load_model,
    load_preprocessor,
    render_sidebar,
)
from utils.prediction import make_predictions
from utils.visualizations import apply_global_styles


st.set_page_config(
    page_title="Batch Prediction",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

st.title("Batch Prediction")
st.caption("Upload customer data, score churn risk, and export retention recommendations.")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV with the same feature columns used during model training.")
    st.stop()

try:
    batch = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

if batch.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

preprocessor = load_preprocessor()

# Save input columns for debugging before feature engineering
input_columns = list(batch.columns)

# Run feature engineering
from utils.feature_engineering import engineer_features
batch = engineer_features(batch)

required_columns = get_training_columns(preprocessor)
if required_columns:
    missing = [column for column in required_columns if column not in batch.columns]
    if missing:
        st.error("The uploaded file is missing required columns after feature engineering: " + ", ".join(missing))
        st.stop()

model_key = get_selected_model_key()
threshold = get_model_threshold(model_key)
model = load_model(model_key)
if model is None:
    st.error("The selected model file is missing or could not be loaded.")
    st.stop()

try:
    predictions = make_predictions(
        batch,
        model,
        preprocessor,
        threshold=threshold,
        allow_fallback=False,
        feature_names=load_feature_names(),
    )
except Exception as exc:
    st.error(f"Prediction failed. Please check the CSV schema and values. Details: {exc}")
    st.stop()

output = predictions.copy()
display = pd.DataFrame(
    {
        "Customer ID": output.get("customerID", output.index.astype(str)),
        "Churn Probability": output["churn_probability"].map(lambda value: f"{value:.1%}"),
        "Predicted Churn": output["predicted_churn"].map({1: "Yes", 0: "No"}),
        "Risk Segment": output["risk_segment"],
        "Recommendation": output["recommendation"],
    }
)

st.success(f"Generated predictions for {len(display):,} customers using threshold {threshold:.2f}.")
st.dataframe(display, width='stretch', hide_index=True)

csv = display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download predictions.csv",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv",
)

# Debug Expander
with st.expander("Debug Features"):
    st.markdown("### Input Columns")
    st.write(input_columns)
    
    st.markdown("### Engineered Columns")
    st.write([c for c in batch.columns if c not in input_columns])
    
    st.markdown("### Final Columns Passed to Model")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        st.write(list(preprocessor.feature_names_in_))
    else:
        st.write("No preprocessor features available")
