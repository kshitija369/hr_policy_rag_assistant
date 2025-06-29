
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    with open("hr_policy.md", "r") as f:
        hr_policy_text = f.read()

    chunks = chunk_text(hr_policy_text)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    vector_store = {i: embeddings[i] for i in range(len(chunks))}

    print("Chunking and embedding complete.")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Shape of embeddings: {embeddings.shape}")
    print("Vector store created in memory.")

if __name__ == "__main__":
    main()
