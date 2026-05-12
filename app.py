import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Deliverable Volume Prediction", layout="wide")

st.title("Stock Deliverable Volume Prediction System")
st.write("Predict current-day deliverable volume and estimated deliverable amount using ADANIPORTS stock data.")

MODEL_PATH = Path("models/stock_deliverable_model.joblib")
FEATURES_PATH = Path("models/features.joblib")
METRICS_PATH = Path("models/metrics.json")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

st.sidebar.header("Project Information")
st.sidebar.write("Target: Deliverable Volume")
st.sidebar.write("Prediction Type: Current-Day Prediction")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write("Note: %Deliverble is not used as input to avoid data leakage.")

if metrics:
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("R² Score", metrics.get("R2 Score", "N/A"))
    st.sidebar.metric("MAE", metrics.get("MAE", "N/A"))
    st.sidebar.metric("RMSE", metrics.get("RMSE", "N/A"))

st.header("Enter Stock Values")

col1, col2, col3 = st.columns(3)

with col1:
    prev_close = st.number_input("Prev Close", min_value=0.0, value=700.0)
    open_price = st.number_input("Open", min_value=0.0, value=705.0)
    high = st.number_input("High", min_value=0.0, value=720.0)

with col2:
    low = st.number_input("Low", min_value=0.0, value=695.0)
    last = st.number_input("Last", min_value=0.0, value=710.0)
    close = st.number_input("Close", min_value=0.0, value=710.0)

with col3:
    vwap = st.number_input("VWAP", min_value=0.0, value=708.0)
    volume = st.number_input("Volume", min_value=0.0, value=1000000.0)
    turnover = st.number_input("Turnover", min_value=0.0, value=708000000.0)

input_data = pd.DataFrame([{
    "Prev Close": prev_close,
    "Open": open_price,
    "High": high,
    "Low": low,
    "Last": last,
    "Close": close,
    "VWAP": vwap,
    "Volume": volume,
    "Turnover": turnover
}])

input_data = input_data[features]

if st.button("Predict Deliverable Volume"):
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)
    estimated_amount = prediction * close

    st.subheader("Prediction Output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted Deliverable Volume", f"{int(prediction):,}")
    with c2:
        st.metric("Estimated Deliverable Amount", f"₹{estimated_amount:,.2f}")

    st.info("Estimated Deliverable Amount = Predicted Deliverable Volume × Close Price")