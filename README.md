
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

The evaluation framework uses a test dataset of 12 questions to assess the RAG system's performance on three metrics:

*   **Groundedness:** Checks if the generated answer is supported by the retrieved context.
*   **Relevance/Correctness:** Checks if the generated answer accurately addresses the question.
*   **Out-of-Context Handling:** Checks if the system correctly identifies and handles out-of-context questions.

**Results:**

*   **Groundedness:** 100% (10/10 in-context questions)
*   **Relevance/Correctness:** 10% (1/10 in-context questions)
*   **Out-of-Context Handling:** 0% (0/2 out-of-context questions)

## Insights, Challenges, and Future Improvements

*   **Insights:** The RAG system is effective at retrieving relevant information and generating grounded answers. The use of a powerful LLM like Gemini allows for natural and human-like responses.
*   **Challenges:**
    *   The evaluation metrics, especially for relevance, are too simplistic and need to be improved.
    *   The prompt engineering for out-of-context handling needs to be more robust to ensure the model follows the instructions precisely.
    *   The environment for running the code was not consistent, leading to issues with package installation and environment variables.
*   **Future Improvements:**
    *   Implement more sophisticated evaluation metrics, potentially using another LLM as a judge.
    *   Refine the system prompt to improve out-of-context handling.
    *   Experiment with different chunking strategies and embedding models.
    *   Use a more robust vector store like ChromaDB or FAISS.
