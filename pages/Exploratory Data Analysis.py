import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Load Data Directly from GitHub ---
GITHUB_CSV_URL = "https://github.com/Nav-Kirat/Islamic_Food_Drive/blob/main/merged_df.xlsx"

@st.cache_data
def load_data():
    return pd.read_csv(GITHUB_CSV_URL)

# Load dataset
merged_df = load_data()

# --- App Title ---
st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Dataset is automatically fetched from GitHub.")

# --- Display Dataset ---
st.write("### 📋 Preview of Dataset")
st.dataframe(merged_df.head())

# --- Sidebar Options ---
st.sidebar.header("📌 Visualization Options")

# --- Numerical Columns ---
numerical_columns = ["age", "dependents_qty", "quantity"]
selected_numerical_col = st.sidebar.selectbox("📊 Select Numerical Column", numerical_columns)

if selected_numerical_col:
    st.subheader(f"📈 Distribution of {selected_numerical_col}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, int(merged_df[selected_numerical_col].max()) + 2) if selected_numerical_col == "dependents_qty" else 20
    sns.histplot(merged_df[selected_numerical_col], kde=True, bins=bins, color="blue", ax=ax)
    plt.title(f"Distribution of {selected_numerical_col}")
    plt.xlabel(selected_numerical_col)
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Boxplot
    st.subheader(f"📦 Boxplot of {selected_numerical_col}")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=merged_df[selected_numerical_col], color="green", ax=ax)
    plt.title(f"Boxplot of {selected_numerical_col}")
    plt.xlabel(selected_numerical_col)
    st.pyplot(fig)

# --- Categorical Columns ---
categorical_columns = ["preferred_languages", "sex", "household", "status", "hamper_type"]
selected_categorical_col = st.sidebar.selectbox("📊 Select Categorical Column", categorical_columns)

if selected_categorical_col:
    st.subheader(f"📊 Distribution of {selected_categorical_col}")

    if selected_categorical_col == "preferred_languages":
        top_n = st.sidebar.slider("Top N Languages", 5, 50, 20)
        top_categories = merged_df["preferred_languages"].value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(y=top_categories.index, x=top_categories.values, palette="viridis", ax=ax)
        plt.title(f"Top {top_n} Preferred Languages")
        plt.xlabel("Count")
        plt.ylabel("Preferred Languages")
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(y=merged_df[selected_categorical_col], order=merged_df[selected_categorical_col].value_counts().index, palette="viridis", ax=ax)
        plt.title(f"Distribution of {selected_categorical_col}")
        plt.xlabel("Count")
        plt.ylabel(selected_categorical_col)
        st.pyplot(fig)
