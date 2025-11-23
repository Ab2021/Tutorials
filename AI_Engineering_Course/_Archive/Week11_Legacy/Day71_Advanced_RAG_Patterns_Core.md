# Day 71: Advanced RAG Patterns
## Core Concepts & Theory

### Beyond Naive RAG

**Naive RAG:** Retrieve top-k chunks -> Stuff in context -> Generate.
**Limitations:**
- **Low Recall:** Misses relevant info if keywords don't match.
- **Lost in Middle:** LLM ignores context in the middle of long prompts.
- **Complex Reasoning:** Cannot answer "Compare X and Y" if X and Y are in different docs.

**Solution:** Advanced RAG Patterns.

### 1. GraphRAG (Knowledge Graphs)

**Concept:**
- Instead of just chunking text, extract entities and relationships.
- **Graph:** Nodes (Entities) and Edges (Relations).
- **Retrieval:** Traverse the graph. "Find all connections between Entity A and Entity B".
- **Benefit:** Captures structural knowledge and multi-hop reasoning.

### 2. RAPTOR (Recursive Abstractive Processing)

**Concept:**
- **Tree Structure:** Cluster chunks -> Summarize clusters -> Cluster summaries -> Summarize...
- **Retrieval:** Search across different levels of abstraction (Tree Traversal).
- **Benefit:** Can answer high-level questions ("What is the main theme?") and low-level details.

### 3. Self-RAG (Self-Reflective RAG)

**Concept:**
- Model critiques its own retrieval and generation.
- **Tokens:** Special tokens `<Retrieve>`, `<IsRel>`, `<IsSup>`, `<IsUse>`.
- **Process:**
  1.  Decide *if* retrieval is needed.
  2.  Retrieve docs.
  3.  Check if docs are *relevant*.
  4.  Generate answer.
  5.  Check if answer is *supported* by docs.

### 4. HyDE (Hypothetical Document Embeddings)

**Concept:**
- Query: "How to bake cake?"
- **Step 1:** LLM generates a *hypothetical* answer: "To bake a cake, you need flour..."
- **Step 2:** Embed the hypothetical answer.
- **Step 3:** Search Vector DB using this embedding.
- **Benefit:** Matches documents that look like the *answer*, not just the *question*.

### 5. Multi-Query & Fusion (RAG Fusion)

**Concept:**
- **Step 1:** Generate 5 variations of the user query.
- **Step 2:** Retrieve docs for all 5 queries.
- **Step 3:** Reciprocal Rank Fusion (RRF) to re-rank and combine results.
- **Benefit:** Higher recall, handles ambiguous queries.

### 6. Contextual Compression & Re-ranking

**Concept:**
- Retrieve 50 docs (high recall).
- Use a Cross-Encoder (Re-ranker) to score them against the query.
- Keep top 5.
- **Compression:** Use an LLM to extract only relevant sentences from the top 5.

### 7. Parent Document Retriever

**Concept:**
- **Index:** Small chunks (sentences).
- **Retrieve:** Small chunks.
- **Return:** The *Parent* chunk (full paragraph/page) to the LLM.
- **Benefit:** Precise search (small chunks) + Rich context (large chunks).

### 8. Corrective RAG (CRAG)

**Concept:**
- Evaluate retrieved documents.
- **Correct:** If docs are good, generate.
- **Ambiguous:** If docs are okay, search web to supplement.
- **Incorrect:** If docs are bad, discard and search web.

### 9. Long Context vs RAG

**Debate:** "Gemini 1.5 has 1M context. Do we need RAG?"
- **Answer:** Yes.
  - **Cost:** 1M tokens is expensive per query.
  - **Latency:** 1M tokens takes seconds/minutes to process.
  - **Dynamic Data:** RAG handles data that changes every minute.
- **Hybrid:** Use RAG to filter 1TB down to 100k tokens, then use Long Context.

### 10. Summary

**RAG Strategy:**
1.  **Structure:** Use **Parent Document Retriever** or **RAPTOR**.
2.  **Query:** Use **HyDE** or **Multi-Query**.
3.  **Refinement:** Always use **Re-ranking**.
4.  **Reasoning:** Use **GraphRAG** for complex relationships.
5.  **Critique:** Use **Self-RAG** logic to verify answers.

### Next Steps
In the Deep Dive, we will implement RAG Fusion with RRF, a HyDE generator, and a simple GraphRAG construction.
