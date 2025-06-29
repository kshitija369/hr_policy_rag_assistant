import json
import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
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
    return [chunks[i] for i in top_k_indices]

def evaluate_groundedness(answer, context):
    # Simple keyword-based check for groundedness
    return any(word in context.lower() for word in answer.lower().split())

def evaluate_relevance(answer, reference_answer):
    # Simple check if the reference answer is contained in the generated answer
    return reference_answer.lower() in answer.lower()

def evaluate_out_of_context(answer):
    return "cannot find information" in answer.lower()

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    with open("hr_policy.md", "r") as f:
        hr_policy_text = f.read()

    chunks = chunk_text(hr_policy_text)

    model_name = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(chunks)
    vector_store = {i: embeddings[i] for i in range(len(chunks))}

    with open("test_questions.json", "r") as f:
        test_data = json.load(f)

    llm = genai.GenerativeModel('gemini-1.5-flash-latest')

    results = []
    for item in test_data:
        question = item["question"]
        expected_type = item["expected_answer_type"]
        reference_answer = item["reference_answer"]

        query_embedding = embedding_model.encode([question])
        relevant_chunks = get_relevant_chunks(query_embedding, vector_store, chunks)
        context = "\n".join(relevant_chunks)

        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        response = llm.generate_content(user_prompt)
        generated_answer = response.text.strip()

        result = {
            "question": question,
            "expected_type": expected_type,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "grounded": False,
            "relevant": False,
            "out_of_context_handled": False
        }

        if expected_type in ["in_context_direct", "in_context_synthesis"]:
            result["grounded"] = evaluate_groundedness(generated_answer, context)
            result["relevant"] = evaluate_relevance(generated_answer, reference_answer)
        elif expected_type == "out_of_context":
            result["out_of_context_handled"] = evaluate_out_of_context(generated_answer)

        results.append(result)

    # Print results
    for res in results:
        print(f"Question: {res['question']}")
        print(f"Generated Answer: {res['generated_answer']}")
        if res['expected_type'] != 'out_of_context':
            print(f"Grounded: {res['grounded']}")
            print(f"Relevant: {res['relevant']}")
        else:
            print(f"Out-of-Context Handled: {res['out_of_context_handled']}")
        print("---")

if __name__ == "__main__":
    main()
