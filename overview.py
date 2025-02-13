import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Food Hamper Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# --- HEADER ---
st.title("📦 Food Hamper Demand Forecasting")
st.subheader("Helping Islamic Family plan resources effectively")

# --- PROBLEM STATEMENT ---
st.markdown("### 🔍 Problem Statement")
st.write("""
Food insecurity remains a significant challenge for many families. Predicting the demand for food hampers 
can help optimize resource allocation and ensure that those in need receive timely assistance.
""")

# --- PROBLEM DEFINITION ---
st.markdown("### 📌 Problem Definition")
st.write("""
The goal of this project is to forecast the number of food hampers required each month. 
By analyzing key factors such as client visit frequency and travel distance, we aim to improve demand prediction accuracy.
""")

# --- OBJECTIVE ---
st.markdown("### 🎯 Objective")
st.write("""
The primary objective is to forecast **monthly** food hamper demand to help **Islamic Family** plan resources effectively.
This will be achieved by aggregating **daily** predictions into monthly forecasts.
""")

# --- PREDICTION TASK ---
st.markdown("### 📊 Prediction Task")
st.write("""
- **Task**: Predict the number of food hampers needed per month.
- **Method**: Aggregate daily predictions into a monthly demand forecast.
""")

# --- KEY INPUTS ---
st.markdown("### 🔑 Key Inputs")
st.write("""
The model utilizes the following factors to make predictions:
- **Client visit frequency** 📅
- **Distance traveled by clients** 🚗
""")

# --- ML TYPE ---
st.markdown("### 🧠 Type of ML Task")
st.write("""
This problem falls under **Time Series Forecasting / Regression**, where past patterns help predict future trends.
""")

# --- FOOTER ---
st.markdown("---")
st.markdown("🚀 **Developed with Streamlit** | 📊 **Powered by Machine Learning**")


