import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Single Date Demand Forecast", page_icon="📅", layout="wide")
st.title("📆 Predict Demand for Any Future Date")

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("daily_hamper_demand_forecast_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
features = model_data["features"]
last_date = model_data["last_date"]
last_vals = model_data["last_values"]

# --- Date Input ---
min_date = last_date + timedelta(days=1)
selected_date = st.date_input("Select a future date", min_value=min_date.date())

# --- Forecast Logic ---
def predict_for_date(target_date):
    days_ahead = (target_date - last_date.date()).days
    future_df = pd.DataFrame({
        "date": [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    })

    # Generate required features
    future_df["day_of_year"] = future_df["date"].dt.dayofyear
    future_df["month"] = future_df["date"].dt.month
    future_df["day_of_week"] = future_df["date"].dt.dayofweek
    future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)

    future_df["day_sin"] = np.sin(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["day_cos"] = np.cos(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["week_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["week_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)

    # Static features
    for col in ["unique_clients", "total_dependents", "returning_proportion"]:
        future_df[col] = last_vals[col]

    recent = list(last_vals["daily_hamper_demand"])
    preds = []

    for i in range(days_ahead):
        lag_1d = recent[-1] if i == 0 else preds[i-1]
        lag_7d = recent[-7] if len(recent) >= 7 else np.mean(recent)
        lag_30d = recent[-30] if len(recent) >= 30 else np.mean(recent)

        roll_7d = np.mean(recent[-7:]) if len(recent) >= 7 else np.mean(recent)
        roll_30d = np.mean(recent[-30:]) if len(recent) >= 30 else np.mean(recent)

        future_df.loc[i, "lag_1d"] = lag_1d
        future_df.loc[i, "lag_7d"] = lag_7d
        future_df.loc[i, "lag_30d"] = lag_30d
        future_df.loc[i, "rolling_mean_7d"] = roll_7d
        future_df.loc[i, "rolling_mean_30d"] = roll_30d
        future_df.loc[i, "rolling_std_7d"] = 0.1 * roll_7d

        X_input = future_df.loc[[i], features]
        pred = model.predict(X_input)[0]
        preds.append(pred)

    # Return the last prediction
    return preds[-1], future_df.iloc[-1]["date"], future_df.iloc[-1]["day_of_week"]

# --- Run Prediction ---
if selected_date:
    pred, pred_date, dow = predict_for_date(selected_date)
    dow_str = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(dow)]

    st.subheader("🔮 Forecast Result")
    st.write(f"**Date:** {pred_date.strftime('%Y-%m-%d')}")
    st.write(f"**Day:** {dow_str}")
    st.metric("📦 Predicted Demand", f"{pred:.0f} hampers")
