# Day 44: Vector DBs in Production

## 1. Beyond the Demo

Running Chroma in-memory is fine for 10 docs.
What about 10 Million docs?

---

## 2. Metadata Filtering

*   **Scenario**: You have documents for User A and User B.
*   **Query**: User A searches "My tax returns".
*   **Risk**: Vector Search might return User B's tax returns (because they are semantically similar).
*   **Solution**: **Metadata Filtering**.
    *   Store `{"user_id": "A"}` with the vector.
    *   Filter: `WHERE user_id = "A"`.

### 2.1 Pre-Filtering vs Post-Filtering
*   **Post-Filtering**: Find top 100 vectors -> Filter for User A. (Risk: Returns 0 results if User A isn't in top 100).
*   **Pre-Filtering**: Filter for User A -> Search within that subset. (Correct approach).

---

## 3. Hybrid Search

Vector search is fuzzy. Sometimes you need exactness.
*   **Keyword Search (BM25)**: Matches "Error 503".
*   **Vector Search**: Matches "Service Unavailable".
*   **Hybrid**: Run both. Combine results using **RRF (Reciprocal Rank Fusion)**.

---

## 4. Scaling

### 4.1 Quantization
*   Vectors are `Float32` (4 bytes per number). 1536 dims = 6KB per vector.
*   1B vectors = 6TB RAM. Expensive.
*   **Quantization**: Compress to `Int8` or `Binary`. 4x-32x smaller. Slightly less accurate.

### 4.2 Sharding
*   Split vectors across multiple nodes.
*   Qdrant/Weaviate handle this automatically.

---

## 5. Summary

Today we hardened the system.
*   **Filter**: Security and relevance.
*   **Hybrid**: Best of both worlds.
*   **Scale**: Quantize and Shard.

**Tomorrow (Day 45)**: We will build the ultimate AI app: **Agents**. Giving the LLM tools to take action.
