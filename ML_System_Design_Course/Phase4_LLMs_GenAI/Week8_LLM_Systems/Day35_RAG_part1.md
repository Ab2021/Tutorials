# Day 35 (Part 1): Advanced RAG Patterns

> **Phase**: 6 - Deep Dive
> **Topic**: Context Engineering
> **Focus**: GraphRAG, HyDE, and Chunking
> **Reading Time**: 60 mins

---

## 1. Advanced Retrieval

### 1.1 Parent Document Retriever
*   **Problem**: Small chunks are good for embedding match, but lack context for LLM.
*   **Solution**: Index small chunks. Retrieve them. Return their *Parent* (Large Chunk) to the LLM.

### 1.2 GraphRAG (Microsoft)
*   **Idea**: Build a Knowledge Graph from docs.
*   **Retrieval**: Traverse graph to find related concepts, not just keyword matches.
*   **Use Case**: "Summarize themes in this dataset." (Global understanding).

---

## 2. Chunking Strategies

### 2.1 Semantic Chunking
*   Don't split by fixed 512 tokens.
*   Split when embedding distance between sentences spikes (Topic shift).

---

## 3. Tricky Interview Questions

### Q1: HyDE (Hypothetical Document Embeddings)?
> **Answer**:
> *   Query: "How to cure flu?"
> *   LLM Generates: "To cure flu, drink water and rest." (Hallucinated but relevant keywords).
> *   Embed the *generation*. Search against real docs.
> *   Matches "semantic intent" better than raw query.

### Q2: RAG vs Long Context (1M tokens)?
> **Answer**:
> *   **Long Context**: "Lost in Middle". Slow. Expensive.
> *   **RAG**: Precise. Cheap. Better for massive knowledge bases (TB scale).

### Q3: How to handle tables in RAG?
> **Answer**:
> *   Convert table to Markdown/JSON.
> *   Or generate a summary of the table and embed that.

---

## 4. Practical Edge Case: Citations
*   **Requirement**: "Answer with [1]".
*   **Fix**: Post-processing. Or constrain generation with Logit Bias.

