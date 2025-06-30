from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_relevant_chunks(query_embedding, vector_store, chunks, k=3):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), list(vector_store.values()))[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]
    return [chunks[i] for i in top_k_indices], top_k_scores

def main():
    with open("hr_policy.md", "r") as f:
        hr_policy_text = f.read()

    chunks = chunk_text(hr_policy_text)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    vector_store = {i: embeddings[i] for i in range(len(chunks))}

    import sys
    if len(sys.argv) > 1:
        user_question = sys.argv[1]
    else:
        user_question = "What is the sick leave policy?"

    query_embedding = model.encode([user_question])

    relevant_chunks, top_k_scores = get_relevant_chunks(query_embedding, vector_store, chunks)

    # Implement Retrieval Score Thresholding
    RETRIEVAL_THRESHOLD = 0.45  # Adjusted threshold
    max_similarity_score = np.max(top_k_scores)

    if max_similarity_score < RETRIEVAL_THRESHOLD:
        print("
--- LLM Response ---")
        print("I am sorry, but I cannot find information on that specific topic in the provided HR documents.")
        return

    system_prompt = """You are an expert HR assistant. Your task is to answer employee questions STRICTLY based on the provided HR policy documents. If the answer is NOT present in the provided context, you MUST state 'I am sorry, but I cannot find information on that specific topic in the provided HR documents.' Do not make up information."""

    user_prompt = f"""Context:
{'\n'.join(relevant_chunks)}

Question: {user_question}

Answer:"""

    import os
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    response = model.generate_content(user_prompt)

    print("
--- LLM Response ---")
    print(response.text)

if __name__ == "__main__":
    main()
