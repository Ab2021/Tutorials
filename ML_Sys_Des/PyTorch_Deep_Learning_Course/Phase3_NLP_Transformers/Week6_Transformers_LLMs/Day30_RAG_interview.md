# Day 30: RAG & LangChain - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Retrieval, Vector DBs, and System Design

### 1. What is the "Context Window" limit?
**Answer:**
*   The maximum number of tokens the LLM can process (e.g., 4k, 32k, 128k).
*   RAG is a workaround to "extend" memory by retrieving only relevant info.

### 2. Why use "Chunking"?
**Answer:**
*   Embedding models have limits (e.g., 512 tokens).
*   Retrieving a whole book for one fact is wasteful and confuses the LLM.
*   Chunks should be semantically self-contained.

### 3. What is "Hybrid Search"?
**Answer:**
*   Combining **Sparse Search** (BM25/Keyword) and **Dense Search** (Vector/Semantic).
*   Keywords are good for exact matches (Part numbers, names).
*   Vectors are good for concepts ("Something to sit on" -> Chair).
*   Combined via Reciprocal Rank Fusion (RRF).

### 4. Explain "HNSW".
**Answer:**
*   Hierarchical Navigable Small World.
*   A graph algorithm for Approximate Nearest Neighbor (ANN) search.
*   Builds a multi-layer graph to allow "skipping" through the vector space quickly.

### 5. What is "ReAct" prompting?
**Answer:**
*   Reasoning + Acting.
*   A pattern where the LLM generates a "Thought", performs an "Action" (Tool call), observes the output, and repeats.

### 6. What is "LangSmith"?
**Answer:**
*   Observability tool for LangChain.
*   Traces the execution of chains/agents.
*   Helps debug costs, latency, and errors.

### 7. What is "Parent Document Retriever"?
**Answer:**
*   Embed small chunks (for accurate matching).
*   Retrieve the *Parent* (larger chunk) of the matched small chunk.
*   Gives the LLM more context while keeping search precise.

### 8. What is "Self-Querying"?
**Answer:**
*   Using the LLM to convert a natural language query into a structured filter.
*   "Sci-fi movies from 1990" $\to$ `filter(genre="sci-fi", year=1990)`.

### 9. What is "Lost in the Middle" phenomenon?
**Answer:**
*   LLMs tend to pay attention to the start and end of the context window, ignoring information in the middle.
*   Re-ranking should place the most relevant chunks at the start/end.

### 10. How do you evaluate RAG?
**Answer:**
*   **RAGAS** (Retrieval Augmented Generation Assessment).
*   Metrics: Faithfulness (Answer matches Context), Answer Relevance (Answer matches Query), Context Precision.

### 11. What is "Embeddings"?
**Answer:**
*   Dense vector representations of text.
*   Captures semantic meaning.
*   OpenAI `text-embedding-3`, `bge-m3`.

### 12. What is "Chain of Thought" (CoT)?
**Answer:**
*   Prompting strategy: "Let's think step by step."
*   Encourages the LLM to generate intermediate reasoning steps, improving accuracy on math/logic.

### 13. What is "Function Calling"?
**Answer:**
*   Fine-tuning LLMs to output structured JSON to call functions.
*   "Get weather in NY" $\to$ `{"func": "get_weather", "args": {"loc": "NY"}}`.

### 14. What is "Multi-Vector Retriever"?
**Answer:**
*   Storing multiple vectors per document (e.g., Summary vector + Full text vector).
*   Decouples what you search (Summary) from what you feed the LLM (Full text).

### 15. What is "GraphRAG"?
**Answer:**
*   Using Knowledge Graphs combined with Vector Search.
*   Retrieves relationships/entities, not just text chunks.
*   Good for multi-hop reasoning.

### 16. Why is "Cosine Similarity" preferred over "L2 Distance"?
**Answer:**
*   L2 is sensitive to magnitude.
*   Cosine is normalized (angle).
*   Most embedding models are trained with Cosine/Dot Product in mind.

### 17. What is "HyDE" (Hypothetical Document Embeddings)?
**Answer:**
*   Query Transformation.
*   LLM generates a *hypothetical* answer to the question.
*   Embed the hypothetical answer and search for that.
*   Matches "Answer-to-Answer" similarity instead of "Question-to-Answer".

### 18. What is "Metadata Filtering"?
**Answer:**
*   Filtering search results based on tags (Date, Author, Source).
*   Pre-filtering (efficient) vs Post-filtering (wasteful).

### 19. What is "Query Expansion"?
**Answer:**
*   Generating synonyms or related questions to broaden the search.
*   Increases Recall.

### 20. What is "Agentic RAG"?
**Answer:**
*   An agent that can plan *how* to retrieve.
*   "Do I need to search Wikipedia or the internal DB? Do I need to search twice?"
