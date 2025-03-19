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
@st.cache_data
def load_data():
    """Loads the dataset from a local Excel file and caches it."""
    return pd.read_excel("merged_df.xlsx")

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

def plot_categorical(column, top_n=20):
    """Plots a bar chart for categorical columns, handling preferred_languages separately."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if column == "preferred_languages":
        top_categories = merged_df[column].value_counts().head(top_n)
        sns.barplot(y=top_categories.index, x=top_categories.values, palette="viridis", ax=ax)
        ax.set_title(f"Top {top_n} Preferred Languages")
    else:
        sns.countplot(y=merged_df[column], order=merged_df[column].value_counts().index, palette="viridis", ax=ax)
        ax.set_title(f"Distribution of {column}")

    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    st.pyplot(fig)

# --- App Title ---
st.title("📊 Exploratory Data Analysis (EDA) Dashboard")

# --- Display Dataset Preview ---
st.write("### 📋 Preview of Dataset")
st.dataframe(merged_df.head())

# --- Sidebar Options ---
st.sidebar.header("📌 Visualization Options")

# --- Dynamically Detect Column Types ---
numerical_columns = merged_df.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = merged_df.select_dtypes(exclude=["number"]).columns.tolist()

# --- Numerical Data Visualization ---
if numerical_columns:
    selected_numerical_col = st.sidebar.selectbox("📊 Select Numerical Column", numerical_columns)

    if selected_numerical_col:
        st.subheader(f"📈 Distribution of {selected_numerical_col}")
        plot_histogram(selected_numerical_col)

        st.subheader(f"📦 Boxplot of {selected_numerical_col}")
        plot_boxplot(selected_numerical_col)

# --- Categorical Data Visualization ---
if categorical_columns:
    selected_categorical_col = st.sidebar.selectbox("📊 Select Categorical Column", categorical_columns)

    if selected_categorical_col:
        st.subheader(f"📊 Distribution of {selected_categorical_col}")

        top_n = st.sidebar.slider("Top N Categories (Only for Preferred Languages)", 5, 50, 20)
        plot_categorical(selected_categorical_col, top_n)
