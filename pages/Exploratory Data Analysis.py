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

# --- Load Dataset ---
def load_data():
    """Loads the dataset and removes unnamed columns."""
    df = pd.read_excel("merged_df.xlsx")  # Load Excel file
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # Remove unnamed columns
    return df

try:
    merged_df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --- Helper Functions ---
def plot_histogram(column):
    """Plots a histogram for the selected numerical column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, int(merged_df[column].max()) + 2) if column == "dependents_qty" else 20
    sns.histplot(merged_df[column], kde=True, bins=bins, color="blue", ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plot_boxplot(column):
    """Plots a boxplot for the selected numerical column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=merged_df[column], color="green", ax=ax)
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel(column)
    st.pyplot(fig)

def plot_categorical(column):
    """Plots a bar chart for categorical columns."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(y=merged_df[column], order=merged_df[column].value_counts().index, palette="viridis", ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    st.pyplot(fig)

# --- App Title ---
st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Dataset is automatically loaded from local storage.")

# --- Display Dataset Preview ---
st.write("### 📋 Preview of Dataset")
st.dataframe(merged_df.head())

# --- Sidebar Options ---
st.sidebar.header("📌 Visualization Options")

# --- Detect Numerical Columns ---
numerical_columns = merged_df.select_dtypes(include=["number"]).columns.tolist()
selected_numerical_col = st.sidebar.selectbox("📊 Select Numerical Column", numerical_columns)

if selected_numerical_col:
    st.subheader(f"📈 Distribution of {selected_numerical_col}")
    plot_histogram(selected_numerical_col)

    st.subheader(f"📦 Boxplot of {selected_numerical_col}")
    plot_boxplot(selected_numerical_col)

# --- Ensure Only "household" and "sex" Appear ---
categorical_columns = [col for col in ["household", "sex"] if col in merged_df.columns]  # Only use existing columns

if not categorical_columns:
    st.sidebar.warning("⚠️ No valid categorical columns found!")
else:
    selected_categorical_col = st.sidebar.selectbox("📊 Select Categorical Column", categorical_columns)

    if selected_categorical_col:
        st.subheader(f"📊 Distribution of {selected_categorical_col}")
        plot_categorical(selected_categorical_col)
