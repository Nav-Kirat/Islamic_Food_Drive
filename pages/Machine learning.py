import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Hamper Demand Forecasting", page_icon="📈", layout="wide")
st.title("📦 Monthly Hamper Demand Forecasting")

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("enriched_dataset.csv")
    df["pickup_month"] = pd.to_datetime(df["pickup_month"], format="%b-%y")
    df = df.sort_values("pickup_month").reset_index(drop=True)
    return df

df = load_data()

# --- Feature Setup ---
feature_cols = [
    'total_visits', 'visits_last_90d', 'days_since_first_visit',
    'avg_days_between_visits', 'distance_km', 'avg_distance_km',
    'total_dependents', 'unique_clients', 'returning_proportion',
    'prev_month_demand', 'rolling_3m_demand'
]
target_col = 'monthly_hamper_demand'
df_model = df.dropna(subset=feature_cols + [target_col])

X = df_model[feature_cols]
y = df_model[target_col]

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", ElasticNet(random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
st.markdown(f"- **R² Score:** `{r2:.4f}`")
st.markdown(f"- **RMSE:** `{rmse:.2f}`")

# --- Forecast Next Month (Default Forecast) ---
latest_features = df_model[feature_cols].iloc[[-1]]
next_month_pred = model.predict(latest_features)[0]
st.subheader("🔮 Forecast")
st.metric(label="Predicted Hampers for Next Month", value=f"{next_month_pred:.0f}")

# --- Custom Date Range Forecast ---
st.subheader("📆 Custom Date Range Forecast")

# Date range selectors
min_date = df["pickup_month"].max() + pd.DateOffset(days=1)
default_end = min_date + pd.DateOffset(days=30)
start_date = st.date_input("Start Date", min_value=min_date.date(), value=min_date.date())
end_date = st.date_input("End Date", min_value=start_date, value=default_end.date())

# Predict function
def predict_demand_range(model, start_date, end_date, seed):
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    predictions = []

    prev_demand = seed['prev_month_demand']
    rolling_values = [prev_demand] * 3  # Seed for rolling mean

    for date in dates:
        features = pd.DataFrame([{
            'total_visits': seed['total_visits'],
            'visits_last_90d': seed['visits_last_90d'],
            'days_since_first_visit': seed['days_since_first_visit'] + (date - df_model["pickup_month"].max()).days,
            'avg_days_between_visits': seed['avg_days_between_visits'],
            'distance_km': seed['distance_km'],
            'avg_distance_km': seed['avg_distance_km'],
            'total_dependents': seed['total_dependents'],
            'unique_clients': seed['unique_clients'],
            'returning_proportion': seed['returning_proportion'],
            'prev_month_demand': prev_demand,
            'rolling_3m_demand': np.mean(rolling_values[-3:])
        }])
        prediction = model.predict(features)[0]
        predictions.append((date, prediction))

        prev_demand = prediction
        rolling_values.append(prediction)

    return pd.DataFrame(predictions, columns=["Date", "Predicted Demand"])

# Prepare seed values from last row
seed_values = {
    'total_visits': latest_features.iloc[0]['total_visits'],
    'visits_last_90d': latest_features.iloc[0]['visits_last_90d'],
    'days_since_first_visit': latest_features.iloc[0]['days_since_first_visit'],
    'avg_days_between_visits': latest_features.iloc[0]['avg_days_between_visits'],
    'distance_km': latest_features.iloc[0]['distance_km'],
    'avg_distance_km': latest_features.iloc[0]['avg_distance_km'],
    'total_dependents': latest_features.iloc[0]['total_dependents'],
    'unique_clients': latest_features.iloc[0]['unique_clients'],
    'returning_proportion': latest_features.iloc[0]['returning_proportion'],
    'prev_month_demand': df_model[target_col].iloc[-1],
    'rolling_3m_demand': df_model[target_col].iloc[-3:].mean()
}

# Run prediction
if start_date and end_date:
    forecast_df = predict_demand_range(model, pd.to_datetime(start_date), pd.to_datetime(end_date), seed_values)

    # Plot forecast
    st.subheader("📈 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df["Date"], forecast_df["Predicted Demand"], marker='o')
    ax.set_title("Forecasted Monthly Hamper Demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hampers")
    ax.grid(True)
    st.pyplot(fig)

    # Show table
    st.subheader("📋 Forecast Table")
    # Ensure it's pandas datetime (even if it's already date)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors='coerce')
    
    # Drop any rows where Date couldn't be parsed (just in case)
    forecast_df = forecast_df.dropna(subset=["Date"])
    
    # Show formatted table
    st.dataframe(forecast_df.assign(Date=forecast_df["Date"].dt.strftime('%Y-%m-%d')))


