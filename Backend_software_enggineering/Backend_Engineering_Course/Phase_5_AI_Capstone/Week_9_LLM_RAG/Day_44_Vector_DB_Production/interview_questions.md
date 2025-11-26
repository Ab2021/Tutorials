# Day 44: Interview Questions & Answers

## Conceptual Questions

### Q1: How does HNSW (Hierarchical Navigable Small World) work?
**Answer:**
*   **Graph Structure**: It builds a multi-layer graph.
    *   **Top Layer**: "Highways". Long jumps across the vector space.
    *   **Bottom Layer**: "Local Roads". Detailed connections.
*   **Search**: Start at top, jump until close, move down, refine.
*   **Speed**: O(log N). Very fast.

### Q2: What is Reciprocal Rank Fusion (RRF)?
**Answer:**
*   **Problem**: How to combine results from Keyword Search (Score 0-100) and Vector Search (Score 0-1)? You can't just add them.
*   **Solution**: Rank them.
    *   Item A: Rank 1 in Keyword, Rank 5 in Vector.
    *   Score = `1/1 + 1/5`.
*   **Benefit**: No need to normalize scores. Works well for Hybrid Search.

### Q3: Why use `pgvector` (Postgres) instead of a dedicated Vector DB?
**Answer:**
*   **Pros**:
    *   **Simplicity**: One DB to manage.
    *   **ACID**: Transactional consistency between data and vectors.
    *   **Joins**: `SELECT * FROM items JOIN embeddings ...`.
*   **Cons**:
    *   **Scale**: Dedicated DBs (Qdrant) scale better to billions of vectors.
    *   **Features**: Dedicated DBs have better quantization/hybrid search support.

---

## Scenario-Based Questions

### Q4: You need to implement "Semantic Search" for an E-Commerce site with 10M products. Which DB do you choose?
**Answer:**
*   **Elasticsearch**: If you already use it, it has vector support now. Good for Hybrid.
*   **Qdrant/Weaviate**: If starting fresh. High performance, open source.
*   **Pinecone**: If you want fully managed (Serverless).

### Q5: Users complain that search results are "outdated". You updated the product description, but the search still shows the old one. Why?
**Answer:**
*   **Cause**: You updated the SQL DB, but didn't update the Vector DB.
*   **Fix**: **CDC (Change Data Capture)**.
    *   Listen to SQL changes.
    *   Trigger a re-embedding job.
    *   Update Vector DB.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to re-embed the entire database every night. Is this a good idea?
**Answer:**
*   **No**.
*   **Cost**: OpenAI Embedding API costs money. 10M items * 1000 tokens = Billions of tokens ($$$).
*   **Load**: Heavy load on DB.
*   **Better**: **Incremental Updates**. Only re-embed items that changed (`updated_at > last_run`).
