import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Hamper Demand Forecasting", page_icon="📈", layout="wide")
st.title("📦 Monthly Hamper Demand Forecasting")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("enriched_dataset.csv")
    df["pickup_month"] = pd.to_datetime(df["pickup_month"], format="%b-%y")
    df = df.sort_values("pickup_month").reset_index(drop=True)
    return df

df = load_data()

# --- Feature Selection ---
feature_cols = [
    'total_visits', 'visits_last_90d', 'days_since_first_visit',
    'avg_days_between_visits', 'distance_km', 'avg_distance_km',
    'total_dependents', 'unique_clients', 'returning_proportion',
    'prev_month_demand', 'rolling_3m_demand'
]
target_col = 'monthly_hamper_demand'

# Drop rows with missing values
df_model = df.dropna(subset=feature_cols + [target_col])
X = df_model[feature_cols]
y = df_model[target_col]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# --- Train Model ---
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

# --- Forecast Next Month ---
latest_features = df_model[feature_cols].iloc[[-1]]
next_month_pred = model.predict(latest_features)[0]

st.subheader("🔮 Forecast")
st.metric(label="Predicted Hampers for Next Month", value=f"{next_month_pred:.0f}")

# --- Plot Actual vs Predicted ---
st.subheader("📈 Actual vs Predicted")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_model["pickup_month"].iloc[-len(y_test):], y_test, label="Actual", marker='o')
ax.plot(df_model["pickup_month"].iloc[-len(y_test):], y_pred, label="Predicted", marker='x', linestyle='--')
ax.set_title("Actual vs Predicted Monthly Hamper Demand")
ax.set_xlabel("Month")
ax.set_ylabel("Hampers")
ax.legend()
ax.grid(True)
st.pyplot(fig)
