# HR Policy RAG Assistant

## Project Description

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about a mock HR policy document. The system uses a sentence-transformer model to generate embeddings for text chunks and a Google Gemini model to generate answers based on the retrieved context.

## Setup and Execution

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key:**

    Set the `GOOGLE_API_KEY` environment variable with your Google AI API key:

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

3.  **Run the HR Assistant:**

    ```bash
    python hr_assistant.py "Your HR-related question"
    ```

4.  **Run the Evaluation:**

    ```bash
    python evaluate_rag.py
    ```

## RAG Pipeline Architecture

1.  **Chunking:** The `hr_policy.md` document is split into overlapping chunks of 200 tokens with a 50-token overlap.
2.  **Embedding:** Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` sentence-transformer model.
3.  **Vector Store:** The embeddings are stored in a simple in-memory Python dictionary.
4.  **Retrieval:** When a user asks a question, the question is embedded using the same model, and a cosine similarity search is performed to find the top 3 most relevant chunks.
5.  **Generation:** The retrieved chunks are combined with the user's question and a system prompt to create a final prompt for the `gemini-1.5-flash-latest` model, which then generates the answer.

## Prompt Engineering Strategy

The system prompt is designed to constrain the LLM to answer only based on the provided context:

```
You are an expert HR assistant. Your task is to answer employee questions STRICTLY based on the provided HR policy documents. If the answer is NOT present in the provided context, you MUST state 'I am sorry, but I cannot find information on that specific topic in the provided HR documents.' Do not make up information.
```

## Evaluation Methodology and Results

The evaluation framework uses a test dataset of 12 questions to assess the RAG system's performance on three metrics, utilizing an LLM (Gemini 1.5 Flash Latest) as a judge for more robust assessment:

*   **Groundedness:** The LLM judge verifies if the generated answer is entirely supported by the retrieved context.
*   **Relevance/Correctness:** The LLM judge assesses if the generated answer accurately and relevantly addresses the original question, given the context and comparing it against a reference answer.
*   **Out-of-Context Handling:** Checks if the system correctly identifies and handles out-of-context questions by stating that information cannot be found.

**Results (using LLM-as-a-Judge with Retrieval Score Thresholding at 0.45):**

*   **Groundedness:** 100.00%
*   **Relevance/Correctness:** 66.67%
*   **Out-of-Context Handling:** 100.00%

## Considerations for LLM-as-a-Judge

*   **Cost & Latency:** Each evaluation step involves an additional LLM call, increasing cost and execution time. For larger datasets, consider optimizing by running evaluation less frequently or using a faster, cheaper LLM for the judge.
*   **Judge's Consistency:** LLMs can be inconsistent. Running the judge multiple times and taking an average, or using few-shot examples within the judge prompt, can improve reliability.
*   **Parsing Output:** Robust parsing of the judge's output is crucial (e.g., looking for keywords like "YES", "NO", ignoring case or surrounding text).
*   **Transparency:** It is important to clearly state that an LLM was used for evaluation when presenting results.

## Insights, Challenges, and Future Improvements

*   **Insights:** The RAG system is highly effective at retrieving relevant information and generating grounded answers. The LLM-as-a-Judge approach provides a more nuanced and accurate evaluation of groundedness and relevance. Retrieval score thresholding significantly improved out-of-context handling.
*   **Challenges:**
    *   Finding the optimal retrieval threshold is crucial; a too-high threshold can filter out relevant in-context questions, while a too-low threshold can lead to hallucinations for out-of-context queries.
    *   Ensuring the consistency and reliability of the LLM judge itself is an ongoing challenge.
*   **Future Improvements:**
    *   Further refine the system prompt for out-of-context handling.
    *   Experiment with different LLM judge prompts and few-shot examples to improve consistency.
    *   Explore more advanced techniques for evaluating LLM responses, such as fine-tuning a smaller model specifically for judging.
    *   Experiment with different chunking strategies and embedding models.
    *   Use a more robust vector store like ChromaDB or FAISS.
