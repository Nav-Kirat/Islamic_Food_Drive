import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# --- Page Config ---
st.set_page_config(page_title="Ask the Assistant", page_icon="💬", layout="wide")
st.title("🤖 Ask the Food Drive Assistant")

# --- Load & Prepare Data ---
@st.cache_data
def load_sample_data():
    data = {
        "Client_ID": [101, 102, 103],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [29, 34, 42],
        "Pickup_Date": ["2023-01-15", "2023-02-15", "2023-03-15"],
        "Hamper_Type": ["Standard", "Premium", "Standard"],
        "Location": ["Downtown", "Uptown", "Midtown"]
    }
    return pd.DataFrame(data)

# Or load real data if needed
transaction_data = load_sample_data()

# --- Create Narrative from Data ---
def generate_narrative(df: pd.DataFrame) -> str:
    narrative = "Here are the latest client transactions:\n"
    for idx, row in df.iterrows():
        narrative += (
            f"Client {row['Client_ID']} ({row['Name']}, Age {row['Age']}) picked "
            f"up a {row['Hamper_Type']} hamper at {row['Location']} on {row['Pickup_Date']}.\n"
        )
    return narrative

transaction_narrative = generate_narrative(transaction_data)

# --- Static Context (Org Description) ---
charity_info = (
    "XYZ Charity is a non-profit organization focused on distributing food hampers. "
    "It aims to improve community well-being by providing support to families in need."
)

# --- Documents for RAG ---
documents = {
    "doc1": charity_info,
    "doc2": transaction_narrative
}

# --- Load Embedding Model ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# --- Embed Documents ---
doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

# --- Retrieval Function ---
def retrieve_context(query, top_k=1):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
        for doc_id, emb in doc_embeddings.items()
    }
    top_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n\n".join(documents[doc_id] for doc_id, _ in top_doc_ids)
    return context

# --- Load FLAN-T5 Model ---
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

generator = load_generator()

# --- Query the LLM with Context ---
def query_llm(query, context):
    prompt = (
        "You have some background info plus transaction data below. "
        "Analyze the context and answer the user’s query clearly and succinctly.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].replace(prompt, "").strip()

# --- Streamlit Chat UI ---
st.markdown("Ask anything about food hamper pickups, clients, or charity mission.")

user_query = st.text_input("💬 What do you want to know?")
if user_query:
    with st.spinner("Thinking..."):
        context = retrieve_context(user_query, top_k=2)
        response = query_llm(user_query, context)

    st.markdown("### 🧠 Assistant's Response")
    st.success(response)

    with st.expander("📄 Retrieved Context"):
        st.text(context)

# --- Optional: Show Table ---
with st.expander("📋 View Transaction Data"):
    st.dataframe(transaction_data)
