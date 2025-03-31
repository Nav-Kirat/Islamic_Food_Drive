import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# --- Page Config ---
st.set_page_config(page_title="Ask the Assistant", page_icon="💬", layout="wide")
st.title("Ask about Food Drive Data!")

# --- Load & Cache Data ---
@st.cache_data
def load_enriched_data():
    return pd.read_csv("enriched_dataset.csv")

df = load_enriched_data()

# --- Generate Narrative from Data ---
def generate_narrative_from_enriched(df, limit=10):
    df = df.sort_values("pickup_month").dropna(subset=["pickup_month", "monthly_hamper_demand"])
    df = df.head(limit)

    narrative = "Here are recent hamper pickup summaries:\n"
    for _, row in df.iterrows():
        narrative += (
            f"In {row['pickup_month']}, approximately {int(row['monthly_hamper_demand'])} hampers were needed "
            f"to serve {int(row['unique_clients'])} clients. "
            f"The average client traveled {row['avg_distance_km']:.1f} km and had {int(row['total_visits'])} visits. "
            f"Households had around {int(row['total_dependents'])} dependents total. "
            f"Returning rate was {row['returning_proportion']:.2%}.\n"
        )
    return narrative

hamper_narrative = generate_narrative_from_enriched(df)

# --- Static Context (Org Description) ---
charity_info = (
    "Islamic Family is a non-profit organization focused on distributing food hampers. "
    "It aims to improve community well-being by providing culturally appropriate food support "
    "to families in need across Edmonton and surrounding areas."
)

documents = {
    "doc1": charity_info,
    "doc2": hamper_narrative
}

# --- Embed Documents ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

# --- Context Retriever ---
def retrieve_context(query, top_k=2):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
        for doc_id, emb in doc_embeddings.items()
    }
    top_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n\n".join(documents[doc_id] for doc_id, _ in top_doc_ids)
    return context

# --- Load Generator (FLAN-T5) ---
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

generator = load_generator()

# --- Prompt Builder & LLM Response ---
def query_llm(query, context):
    prompt = (
        "You have background information and transaction data below. "
        "Answer the user's query clearly and accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].replace(prompt, "").strip()

# --- UI: User Input & Display ---
st.markdown("Ask about recent food hamper activity, client patterns, or organizational details.")

query = st.text_input("💬 Ask a question:")
if query:
    with st.spinner("Retrieving info and generating answer..."):
        context = retrieve_context(query, top_k=2)
        answer = query_llm(query, context)

    st.markdown("### 🧠 Assistant's Response")
    st.success(answer)

    with st.expander("📄 Retrieved Context"):
        st.text(context)

# --- Optional: Show Raw Data ---
with st.expander("📊 View Data Sample"):
    st.dataframe(df.head(10))
