import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load cleaned complaint data
df = pd.read_csv("cleaned_complaints.csv")
df = df.dropna(subset=["cleaned_narrative"])

# Chunk the narratives
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = []
metadatas = []

for _, row in df.iterrows():
    complaint_id = row["Complaint ID"]
    product = row["Product"]
    text = row["cleaned_narrative"]

    for chunk in text_splitter.split_text(text):
        chunks.append(chunk)
        metadatas.append({
            "complaint_id": complaint_id,
            "product": product,
        })

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)

# Create FAISS index
embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.idx")

with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)
