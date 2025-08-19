import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# ✅ Load Vector Index and Metadata
index = faiss.read_index("vector_store/faiss_index.idx")
with open("vector_store/metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load Cleaned Data
df = pd.read_csv("cleaned_complaints.csv")

# ✅ Load Generator (LLM) — using HuggingFace's small model for demo
generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)

# ✅ Function: Retrieve Top-k Similar Chunks
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

# ✅ Prompt Template
def build_prompt(question, retrieved_chunks):
    context = "\n\n".join([chunk for chunk, meta in retrieved_chunks])
    prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return prompt

# ✅ Full RAG QA Pipeline
def rag_answer(question, k=5):
    retrieved_chunks = retrieve_chunks(question, k)
    prompt = build_prompt(question, retrieved_chunks)
    generated = generator(prompt)[0]["generated_text"]
    answer = generated.split("Answer:")[-1].strip()
    return answer, retrieved_chunks

# ✅ Example Usage
sample_question = "What issues do customers report about credit reporting?"
answer, sources = rag_answer(sample_question, k=5)

print("🔹 Question:", sample_question)
print("🧠 Answer:\n", answer)
print("\n📚 Retrieved Chunks:")
for i, (chunk, meta) in enumerate(sources):
    print(f"\nChunk {i+1} — Complaint ID: {meta['complaint_id']}")
    print(chunk)
