# Placeholder for app.pyimport streamlit as st
import streamlit as st
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector store
index = faiss.read_index("vector_store/faiss_index.idx")
with open("vector_store/metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Load cleaned data and embedding model
df = pd.read_csv("cleaned_complaints.csv")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Use lightweight generator model for speed
generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)

# Retrieve relevant chunks
def retrieve_chunks(question, k=5):
    question_embedding = embedding_model.encode([question])
    D, I = index.search(question_embedding, k)
    retrieved_chunks = []
    for idx in I[0]:
        if idx < len(df):
            chunk = df.iloc[idx]["cleaned_narrative"]
            metadata = metadatas[idx]
            retrieved_chunks.append((chunk, metadata))
    return retrieved_chunks

# Prompt builder
def build_prompt(question, chunks):
    context = "\n\n".join([chunk for chunk, meta in chunks])
    prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return prompt

# RAG answer pipeline
def rag_answer(question, k=5):
    retrieved_chunks = retrieve_chunks(question, k)
    prompt = build_prompt(question, retrieved_chunks)
    generated = generator(prompt)[0]["generated_text"]
    answer = generated.split("Answer:")[-1].strip()
    return answer, retrieved_chunks

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📊 CrediTrust Consumer Complaint Assistant")
st.markdown("Ask any question related to consumer complaints. The AI will search real complaint data and generate a helpful answer.")

# Text input
user_question = st.text_input("Enter your question:", "")

col1, col2 = st.columns([1, 1])
with col1:
    ask_button = st.button("💬 Ask")
with col2:
    clear_button = st.button("🧹 Clear")

# Handle user input
if ask_button and user_question:
    with st.spinner("🔍 Thinking..."):
        answer, sources = rag_answer(user_question)

    st.subheader("🧠 Answer")
    st.success(answer)

    st.subheader("📚 Sources")
    for i, (chunk, meta) in enumerate(sources):
        st.markdown(f"**Chunk {i+1}** — Complaint ID: `{meta['complaint_id']}`")
        st.code(chunk, language="markdown")

if clear_button:
    st.experimental_rerun()
