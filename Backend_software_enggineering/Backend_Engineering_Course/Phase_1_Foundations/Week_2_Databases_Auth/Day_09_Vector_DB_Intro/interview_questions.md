# Day 9: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between KNN (K-Nearest Neighbors) and ANN (Approximate Nearest Neighbors)?
**Answer:**
*   **KNN**: Exact search. Compares query against *every* vector in the DB.
    *   *Accuracy*: 100%.
    *   *Speed*: Slow (`O(N)`).
*   **ANN**: Approximate search (using HNSW/IVF). Uses an index to skip most vectors.
    *   *Accuracy*: ~99% (Recall).
    *   *Speed*: Fast (`O(log N)`).
*   *Context*: Vector DBs almost always use ANN for scale.

### Q2: Why is "Metadata Filtering" hard in Vector DBs?
**Answer:**
*   **Problem**: If you want "Vectors similar to 'Apple' BUT only from 'Category: Fruit'".
    *   *Post-filtering*: Find top 100 matches, then filter by Fruit. Risk: You might filter out all 100 and return nothing.
    *   *Pre-filtering*: Filter by Fruit first, then search. Risk: The index might not support this efficiently.
*   **Solution**: Modern DBs (Qdrant/Weaviate) use "Filtered ANN" (modifying the HNSW graph traversal to only visit nodes matching the filter).

### Q3: What is the "Curse of Dimensionality"?
**Answer:**
*   As the number of dimensions increases (e.g., 1536 dims for OpenAI), the volume of the space increases exponentially.
*   Points become sparse.
*   Distance metrics lose meaning (everything is roughly equidistant).
*   *Impact*: Indexing becomes harder and requires more RAM.

---

## Scenario-Based Questions

### Q4: You are building a Legal Search engine. Accuracy is paramount. Do you use HNSW or Flat Index?
**Answer:**
**Flat Index (Brute Force).**
*   **Reasoning**: In legal search, missing a relevant precedent (Recall < 100%) is unacceptable.
*   **Trade-off**: It will be slow.
*   **Mitigation**: If the dataset is huge, we might use a two-stage approach: Use ANN to get top 1000 candidates, then Rerank them using a Cross-Encoder (more accurate model) or Brute Force check.

### Q5: How do you handle updates in a Vector DB?
**Answer:**
*   **Challenge**: HNSW graphs are complex to update. Inserting a node requires rewiring connections.
*   **Mechanism**: Most Vector DBs use an LSM-tree like structure. New writes go to a "mutable" segment (in-memory). Older data is in "immutable" segments on disk. Background processes merge them.
*   **Impact**: There might be a slight delay (seconds) before a new vector is searchable (Eventual Consistency).

---

## Behavioral / Role-Specific Questions

### Q6: A Product Manager asks why we can't just use Postgres `LIKE` search for the chatbot.
**Answer:**
*   **Explanation**: `LIKE` only matches exact substrings.
    *   User asks: "How do I reset my password?"
    *   Doc says: "To recover your credentials..."
    *   `LIKE` fails.
*   **Vector Search**: Matches "reset password" with "recover credentials" because they are semantically close in the embedding space.
*   **Hybrid**: I would suggest a **Hybrid Search** (Keyword + Vector) to get the best of both worlds (exact matches for product names, semantic for concepts).
