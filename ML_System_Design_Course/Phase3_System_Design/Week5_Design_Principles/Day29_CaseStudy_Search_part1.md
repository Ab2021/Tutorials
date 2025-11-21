# Day 29 (Part 1): Advanced Search Algorithms

> **Phase**: 6 - Deep Dive
> **Topic**: Needle in a Haystack
> **Focus**: HNSW, LSH, and Query Expansion
> **Reading Time**: 60 mins

---

## 1. HNSW Internals

How to build the graph?

### 1.1 Construction
*   Insert node.
*   Find neighbors in Top Layer.
*   Descend. Find neighbors in Lower Layer.
*   **Heuristic**: Keep connections to "diverse" neighbors (not just closest) to ensure graph connectivity (Small World property).

---

## 2. Locality Sensitive Hashing (LSH)

The Old School ANN.

### 2.1 Random Projections (SimHash)
*   Hyperplane cuts space.
*   Points on same side get bit 1, else 0.
*   Repeat K times -> K-bit hash.
*   **Hamming Distance** in hash space $\approx$ Cosine Distance in vector space.

---

## 3. Query Expansion

### 3.1 Pseudo-Relevance Feedback (PRF)
1.  Run Query.
2.  Assume Top 5 docs are relevant.
3.  Extract keywords from Top 5 docs.
4.  Add to Query. Run again.
*   **Risk**: Drift if Top 5 were wrong.

---

## 4. Tricky Interview Questions

### Q1: Precision vs Recall in Search?
> **Answer**:
> *   **Precision**: How many of returned docs are relevant? (User happiness).
> *   **Recall**: How many of relevant docs were returned? (Legal discovery).
> *   **F1**: Harmonic mean.

### Q2: How to optimize "Phrase Search" ("New York")?
> **Answer**:
> *   **Next Word Index**: Store position of words. Check if `pos(York) == pos(New) + 1`.
> *   **N-Grams**: Index "New York" as a token.

### Q3: Explain "Sharding" in Search.
> **Answer**:
> *   **Document Sharding**: Split docs across nodes. Query all nodes. Merge results. (Good for Recall).
> *   **Term Sharding**: Store "A-M" on Node 1, "N-Z" on Node 2. (Bad for latency, requires multi-node coordination for one query).

---

## 5. Practical Edge Case: Stop Words
*   **Old**: Remove "the", "and".
*   **New (BERT)**: Keep them. "To be or not to be" is all stop words but has deep meaning. Contextual embeddings handle them fine.

