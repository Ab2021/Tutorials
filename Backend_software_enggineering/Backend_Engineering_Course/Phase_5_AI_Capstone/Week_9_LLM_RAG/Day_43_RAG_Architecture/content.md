# Day 43: RAG Architecture

## 1. The Knowledge Gap

LLMs are frozen in time (GPT-4 training data ends in 2023). They don't know your private company data.
*   **Fine-Tuning**: Expensive, slow, hard to update.
*   **RAG (Retrieval Augmented Generation)**: Cheap, fast, real-time.

---

## 2. The Pipeline

### 2.1 Ingestion (Load & Split)
*   **Loader**: Read `policy.pdf`.
*   **Splitter**: You can't feed a 100-page PDF to an LLM (Context Window limit).
    *   **RecursiveCharacterTextSplitter**: Splits by paragraphs, then sentences. Keeps context together.
    *   **Chunk Size**: e.g., 1000 characters.
    *   **Overlap**: e.g., 200 characters. Ensures words at the edge aren't cut off.

### 2.2 Indexing (Embed & Store)
*   **Embedding Model**: Converts text chunk -> Vector (`[0.1, -0.5, ...]`).
*   **Vector Store**: Saves the vector + original text.

### 2.3 Retrieval (Search)
*   **Query**: "What is the vacation policy?"
*   **Search**: Find top 3 chunks most similar to the query vector.

### 2.4 Generation (Answer)
*   **Prompt**:
    ```text
    Context: {chunk_1} ... {chunk_3}
    Question: What is the vacation policy?
    Answer using ONLY the context.
    ```
*   **Result**: "The policy states 20 days per year."

---

## 3. Tools

*   **LangChain**: Orchestrates the flow.
*   **ChromaDB / FAISS**: Local Vector Stores (good for dev).
*   **Pinecone / Weaviate**: Cloud Vector Stores (good for prod).

---

## 4. Summary

Today we gave the AI a library card.
*   **Split**: Break it down.
*   **Embed**: Turn it into math.
*   **Retrieve**: Find the needle.
*   **Generate**: Write the answer.

**Tomorrow (Day 44)**: We will take RAG to production. How to handle **Metadata Filtering** and **Scaling**.
