import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Daily Demand Forecast", page_icon="📅", layout="wide")
st.title("📆 Daily Hamper Demand Forecast")

# --- Load Trained Model ---
@st.cache_resource
def load_model():
    with open("daily_hamper_demand_forecast_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
features = model_data["features"]
last_date = model_data["last_date"]
last_vals = model_data["last_values"]

# --- Forecast Function ---
def forecast_daily_demand(days=30):
    future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]
    future_df = pd.DataFrame({"date": future_dates})

    # Date features
    future_df["day_of_year"] = future_df["date"].dt.dayofyear
    future_df["month"] = future_df["date"].dt.month
    future_df["day_of_week"] = future_df["date"].dt.dayofweek
    future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encodings
    future_df["day_sin"] = np.sin(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["day_cos"] = np.cos(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["week_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["week_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)

    # Static features
    for key in ["unique_clients", "total_dependents", "returning_proportion"]:
        future_df[key] = last_vals[key]

    last_30 = list(last_vals["daily_hamper_demand"])

    preds = []
    for i in range(len(future_df)):
        lag_1d = last_30[-1] if i == 0 else preds[-1]
        lag_7d = last_30[-7] if len(last_30) >= 7 else np.mean(last_30)
        lag_30d = last_30[-30] if len(last_30) >= 30 else np.mean(last_30)

        roll_7d = np.mean(last_30[-7:])
        roll_30d = np.mean(last_30[-30:])

        future_df.loc[i, "lag_1d"] = lag_1d
        future_df.loc[i, "lag_7d"] = lag_7d
        future_df.loc[i, "lag_30d"] = lag_30d
        future_df.loc[i, "rolling_mean_7d"] = roll_7d
        future_df.loc[i, "rolling_mean_30d"] = roll_30d
        future_df.loc[i, "rolling_std_7d"] = 0.1 * roll_7d  # optional noise

        pred = model.predict(future_df.loc[[i], features])[0]
        preds.append(pred)

    future_df["predicted_demand"] = preds
    return future_df

# --- Run Forecast ---
forecast_days = st.slider("Days to forecast", 7, 60, 30)
forecast = forecast_daily_demand(forecast_days)

# --- Plot Forecast ---
st.subheader("📈 Forecasted Daily Demand")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast["date"], forecast["predicted_demand"], marker="o", label="Predicted")
ax.set_title("Forecasted Daily Hamper Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Show Table ---
st.subheader("📋 Forecast Table")
st.dataframe(forecast[["date", "predicted_demand"]].rename(columns={
    "date": "Date",
    "predicted_demand": "Expected Demand"
}).assign(Date=lambda df: df["Date"].dt.strftime("%Y-%m-%d")))
