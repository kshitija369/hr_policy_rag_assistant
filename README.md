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

3.  **Run the HR Assistant (Python):**

    ```bash
    python hr_assistant.py "Your HR-related question"
    ```

4.  **Run the Evaluation (Python):**

    ```bash
    python evaluate_rag.py
    ```

5.  **Containerization (Docker):**

    **Build the Docker Image:**

    ```bash
    docker build -t hr-rag-assistant .
    ```

    **Run the HR Assistant (Docker):**

    ```bash
    docker run -e GOOGLE_API_KEY="YOUR_API_KEY" hr-rag-assistant python hr_assistant.py "How much vacation time do I get?"
    ```

    **Run the Evaluation (Docker):**

    ```bash
    docker run -e GOOGLE_API_KEY="YOUR_API_KEY" hr-rag-assistant python evaluate_rag.py
    ```

## RAG Pipeline Architecture

1.  **Chunking:** The `hr_policy.md` document is split into overlapping chunks of 200 tokens with a 50-token overlap.
2.  **Embedding:** Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` sentence-transformer model. This model can be optionally quantized for reduced size and faster inference.
3.  **Vector Store:** The embeddings are stored in a simple in-memory Python dictionary.
4.  **Retrieval:** When a user asks a question, the question is embedded using the same model (original or quantized), and a cosine similarity search is performed to find the top 3 most relevant chunks.
5.  **Generation:** The retrieved chunks are combined with the user's question and a system prompt to create a final prompt for the `gemini-1.5-flash-latest` model, which then generates the answer.

## Quantization of Embedding Model

To optimize the embedding model for deployment, dynamic quantization was applied using ONNX Runtime. This process converts the model's weights and activations to a lower precision (e.g., 8-bit integers) without significant loss in accuracy, leading to a smaller model size and faster inference times.

-   **Process:** The `quantize_embedder.py` script handles the conversion of the `SentenceTransformer` model to ONNX format and then applies dynamic quantization.
-   **Usage:** The `evaluate_rag.py` script now supports an `--quantized` flag to use the quantized embedding model for evaluation. This allows for direct comparison of performance metrics between the original and quantized versions.
-   **Benefits:** As shown in the evaluation results, quantization significantly reduces the model's footprint and improves embedding speed, making the RAG system more efficient for production environments.

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

**Embedding Model Performance Comparison:**

| Metric                     | Original Model | Quantized Model |
|----------------------------|----------------|-----------------|
| Model Size                 | 87.57 MB       | 56.00 MB        |
| Embedding Time (per batch) | 0.2305 seconds | 0.1738 seconds  |
| Groundedness Score         | 100.00%        | 100.00%         |
| Relevance/Correctness Score| 55.56%         | 77.78%          |
| Out-of-Context Handling    | 100.00%        | 100.00%         |

**Key Findings:**
- **Model Size Reduction:** The quantized model achieved a 36.05% reduction in size compared to the original model.
- **Speedup:** The quantized model showed a 24.60% speedup in embedding time.
- **Improved Relevance:** The quantized model significantly improved the Relevance/Correctness score from 55.56% to 77.78%, indicating better retrieval of relevant information.
- **Maintained Performance:** Groundedness and Out-of-Context Handling scores remained at 100% for both models.

**Results (using LLM-as-a-Judge with Retrieval Score Thresholding at 0.45):**

*   **Groundedness:** 100.00%
*   **Relevance/Correctness:** 77.78%
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