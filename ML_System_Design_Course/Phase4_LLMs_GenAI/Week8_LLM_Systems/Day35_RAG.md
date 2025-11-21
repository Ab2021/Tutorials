# Day 35: Retrieval Augmented Generation (RAG)

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: Giving LLMs Memory
> **Reading Time**: 50 mins

---

## 1. The Architecture

LLMs are frozen in time. RAG connects them to live data.

### 1.1 Naive RAG
1.  **Ingest**: Chunk documents -> Embed (OpenAI/Cohere) -> Store in Vector DB.
2.  **Retrieve**: User Query -> Embed -> Search Top K chunks.
3.  **Generate**: Prompt = "Context: {chunks}. Question: {query}". LLM answers.

### 1.2 Advanced RAG (2025 Standard)
*   **Hybrid Search**: Keyword (BM25) + Semantic (Vector).
*   **Re-Ranking**: Retrieve Top 50 chunks. Use a Cross-Encoder (Cohere Rerank) to sort them by relevance. Pass Top 5 to LLM.
*   **Query Transformation**:
    *   *HyDE*: Generate a fake answer, then embed that.
    *   *Multi-Query*: Break complex question into 3 sub-questions.

---

## 2. Real-World Challenges & Solutions

### Challenge 1: The "Lost in the Middle" Phenomenon
**Scenario**: You pass 10 chunks. The answer is in chunk #5. The LLM ignores it.
**Theory**: LLMs pay attention to the start and end of the context window, but often gloss over the middle.
**Solution**:
*   **Re-Ranking**: Ensure the most relevant chunk is at the start or end.
*   **Context Compression**: Summarize chunks before passing them.

### Challenge 2: Bad Retrieval
**Scenario**: User asks "Apple". Vector DB returns "Apple Pie" recipes. User meant "Apple Inc".
**Solution**:
*   **Metadata Filtering**: Filter by `category="tech"`.
*   **Hybrid Search**: BM25 helps match exact terms like error codes or product IDs that vectors miss.

---

## 3. Interview Preparation

### Conceptual Questions

**Q1: Why use RAG instead of Fine-Tuning for knowledge injection?**
> **Answer**:
> 1.  **Freshness**: RAG updates instantly (just add to DB). Fine-tuning takes days.
> 2.  **Hallucination**: RAG allows citing sources. Fine-tuning "bakes" knowledge in, making it hard to verify or update.
> 3.  **Privacy**: Different users can have access to different document sets (RBAC) in RAG. Fine-tuning leaks all data to everyone.

**Q2: What is a Cross-Encoder vs. Bi-Encoder?**
> **Answer**:
> *   **Bi-Encoder (Vector Search)**: Embed Query and Doc independently. Fast ($O(N)$). Good for retrieval.
> *   **Cross-Encoder (Re-Ranker)**: Feed (Query, Doc) together into BERT. Outputs a score. Slow ($O(N)$ BERT passes). Very accurate.
> *   **Pipeline**: Bi-Encoder to get Top 50 -> Cross-Encoder to get Top 5.

**Q3: How do you evaluate a RAG system?**
> **Answer**: Frameworks like **RAGAS**.
> *   **Retrieval Metrics**: Precision@K, Recall@K.
> *   **Generation Metrics**: Faithfulness (Did answer come from context?), Answer Relevance.

---

## 5. Further Reading
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Techniques (LangChain)](https://python.langchain.com/docs/use_cases/question_answering/)
