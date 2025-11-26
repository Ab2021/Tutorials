# Day 28: Vector DB Deep Dive & Hybrid Search

## 1. The AI Revolution

Traditional DBs search for *exact matches* (`WHERE text LIKE '%apple%'`).
AI needs *semantic matches* ("fruit" should match "apple").

### 1.1 Embeddings
The magic sauce. A model (like OpenAI `text-embedding-3-small` or `all-MiniLM-L6-v2`) converts text into a list of numbers (vector).
*   "Dog" -> `[0.1, 0.5, -0.2]`
*   "Puppy" -> `[0.1, 0.55, -0.1]` (Close!)
*   "Car" -> `[0.9, -0.1, 0.0]` (Far away).

---

## 2. Vector Databases (Qdrant, Pinecone, pgvector)

Storing vectors is easy. Searching them is hard.
If you have 1M vectors, calculating distance to all of them (Brute Force / KNN) is too slow.

### 2.1 ANN (Approximate Nearest Neighbor)
We trade 1% accuracy for 100x speed.
*   **HNSW (Hierarchical Navigable Small World)**: The standard algorithm. Think of it like a multi-layer graph (highway vs local roads).
*   **IVF (Inverted File)**: Clustering vectors into buckets.

### 2.2 Distance Metrics
*   **Cosine Similarity**: Measures angle. (Most common for text).
*   **Euclidean (L2)**: Measures straight-line distance.
*   **Dot Product**: Fast, but requires normalized vectors.

---

## 3. Hybrid Search (The Best of Both Worlds)

Vector search has a flaw: It's bad at exact keywords.
*   Query: "iPhone 15 Pro Max"
*   Vector: Might return "Samsung Galaxy S24" (because they are semantically similar phones).
*   Keyword: Returns "iPhone 15 Pro Max".

**Solution**: Combine them.
1.  **Dense Retrieval**: Vector Search.
2.  **Sparse Retrieval**: BM25 (Keyword Search).
3.  **Reciprocal Rank Fusion (RRF)**: Merge the two lists.

---

## 4. RAG (Retrieval Augmented Generation)

The killer app for Vector DBs.
1.  **User**: "How do I reset my password?"
2.  **App**: Search Vector DB for "reset password" docs.
3.  **App**: Send the *relevant docs* + *user question* to ChatGPT.
4.  **ChatGPT**: Answers using the docs.

---

## 5. Summary

Today we gave our database a brain.
*   **Embeddings**: Meaning as numbers.
*   **HNSW**: Fast search.
*   **Hybrid**: Precision + Recall.

**Tomorrow (Day 29)**: We go back to infrastructure. How do we scale our databases to handle 100k writes per second? **Sharding and Replication**.
