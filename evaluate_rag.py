import json
import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import onnxruntime as rt

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

def evaluate_groundedness_llm(llm_judge, context, rag_system_answer):
    system_prompt = """You are an expert evaluator of AI-generated content. Your task is to determine if an 'Answer' is entirely supported by the 'Context' provided. Respond with 'YES' if the answer is fully supported by the context, and 'NO' if any part of the answer is not supported or introduces new information not found in the context."""
    user_prompt = f"""Context:\n{context}\n\nAnswer: {rag_system_answer}\n\nIs the Answer fully supported by the Context? (YES/NO)"""

    response = llm_judge.generate_content([system_prompt, user_prompt])
    return response.text.strip().upper() == "YES"

def evaluate_relevance_llm(llm_judge, question, context, rag_system_answer, reference_answer):
    system_prompt = """You are an expert at comparing answers. Your task is to assess if the 'Generated Answer' conveys the same information and is as complete as the 'Reference Answer', given the 'Context' and 'Question'. Respond with 'YES' if they are largely equivalent in meaning and completeness, and 'NO' otherwise."""
    user_prompt = f"""Question: {question}\n\nContext:\n{context}\n\nReference Answer: {reference_answer}\n\nGenerated Answer: {rag_system_answer}\n\nAre the Generated Answer and Reference Answer equivalent in meaning and completeness based on the Context? (YES/NO)"""

    response = llm_judge.generate_content([system_prompt, user_prompt])
    return response.text.strip().upper() == "YES"

def evaluate_out_of_context(answer):
    return "cannot find information" in answer.lower() or "not contain information" in answer.lower() or "does not contain information" in answer.lower()

class QuantizedEmbedder:
    def __init__(self, model_path):
        self.session = rt.InferenceSession(model_path)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = self.sbert_model.tokenizer # Access the underlying tokenizer

    def encode(self, texts):
        # Tokenize the texts
        # Tokenize the texts
        tokenized_texts = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors='np',
            max_length=self.tokenizer.max_len_single_sentence # Use the tokenizer's max length
        )
        
        # Prepare ONNX input
        onnx_inputs = {
            'input_ids': tokenized_texts['input_ids'].astype(np.int64),
            'attention_mask': tokenized_texts['attention_mask'].astype(np.int64),
            
        }
        
        # Run ONNX inference
        outputs = self.session.run(None, onnx_inputs)
        
        # Extract embeddings (assuming the first output is the embeddings)
        embeddings = outputs[0]
        # Perform mean pooling (similar to Sentence-BERT)
        input_mask_expanded = np.expand_dims(tokenized_texts['attention_mask'], -1).astype(float)
        sum_embeddings = np.sum(embeddings * input_mask_expanded, 1)
        sum_mask = np.maximum(np.sum(input_mask_expanded, 1), 1e-9) # Avoid division by zero
        embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Add a small epsilon to prevent division by zero
        embeddings = embeddings / (norm + 1e-12)
        
        return embeddings

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system with different embedding models.")
    parser.add_argument("--quantized", action="store_true", help="Use quantized embedding model.")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key);

    with open("hr_policy.md", "r") as f:
        hr_policy_text = f.read()

    chunks = chunk_text(hr_policy_text)

    if args.quantized:
        print("Using quantized embedding model...")
        embedding_model = QuantizedEmbedder("quantized_embedder_model.onnx")
    else:
        print("Using original SentenceTransformer model...")
        model_name = 'all-MiniLM-L6-v2'
        embedding_model = SentenceTransformer(model_name)
    
    embeddings = embedding_model.encode(chunks)
    vector_store = {i: embeddings[i] for i in range(len(chunks))}

    with open("test_questions.json", "r") as f:
        test_data = json.load(f)

    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    llm_judge = genai.GenerativeModel('gemini-1.5-flash-latest') # Using the same model for judging

    results = []
    for item in test_data:
        question = item["question"]
        expected_type = item["expected_answer_type"]
        reference_answer = item["reference_answer"]

        query_embedding = embedding_model.encode([question])
        relevant_chunks, top_k_scores = get_relevant_chunks(query_embedding, vector_store, chunks)
        context = "\n".join(relevant_chunks)

        # Retrieval Score Thresholding
        RETRIEVAL_THRESHOLD = 0.45  # Same threshold as in hr_assistant.py
        max_similarity_score = np.max(top_k_scores)

        generated_answer = ""
        if max_similarity_score < RETRIEVAL_THRESHOLD:
            generated_answer = "I am sorry, but I cannot find information on that specific topic in the provided HR documents."
        else:
            user_prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            response = llm.generate_content(user_prompt)
            generated_answer = response.text.strip()

        result = {
            "question": question,
            "expected_type": expected_type,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "grounded": "N/A",
            "relevant": "N/A",
            "out_of_context_handled": "N/A"
        }

        if expected_type in ["in_context_direct", "in_context_synthesis"]:
            result["grounded"] = evaluate_groundedness_llm(llm_judge, context, generated_answer)
            result["relevant"] = evaluate_relevance_llm(llm_judge, question, context, generated_answer, reference_answer)
        elif expected_type == "out_of_context":
            result["out_of_context_handled"] = evaluate_out_of_context(generated_answer)

        result["max_similarity_score"] = max_similarity_score # Add similarity score to results
        results.append(result)

    # Calculate and print overall metrics
    total_in_context = sum(1 for r in results if r["expected_type"] in ["in_context_direct", "in_context_synthesis"])
    total_out_of_context = sum(1 for r in results if r["expected_type"] == "out_of_context")

    correct_grounded = sum(1 for r in results if r["expected_type"] in ["in_context_direct", "in_context_synthesis"] and r["grounded"])
    correct_relevant = sum(1 for r in results if r["expected_type"] in ["in_context_direct", "in_context_synthesis"] and r["relevant"])
    correct_out_of_context_handled = sum(1 for r in results if r["expected_type"] == "out_of_context" and r["out_of_context_handled"])

    groundedness_score = (correct_grounded / total_in_context) * 100 if total_in_context > 0 else 0
    relevance_score = (correct_relevant / total_in_context) * 100 if total_in_context > 0 else 0
    out_of_context_score = (correct_out_of_context_handled / total_out_of_context) * 100 if total_out_of_context > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Groundedness Score (in-context questions): {groundedness_score:.2f}%")
    print(f"Relevance/Correctness Score (in-context questions): {relevance_score:.2f}%")
    print(f"Out-of-Context Handling Score: {out_of_context_score:.2f}%")
    print("\n--- Detailed Results ---")

    for res in results:
        print(f"Question: {res['question']}")
        print(f"Generated Answer: {res['generated_answer']}")
        print(f"Max Similarity Score: {res['max_similarity_score']:.4f}") # Print similarity score
        if res['expected_type'] != 'out_of_context':
            print(f"Grounded (LLM Judge): {res['grounded']}")
            print(f"Relevant (LLM Judge): {res['relevant']}")
        else:
            print(f"Out-of-Context Handled: {res['out_of_context_handled']}")
        print("---")

if __name__ == "__main__":
    main()