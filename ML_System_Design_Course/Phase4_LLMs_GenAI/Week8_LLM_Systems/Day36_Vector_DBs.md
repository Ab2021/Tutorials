# Day 36: Vector Databases & Search

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: The Long-Term Memory of AI
> **Reading Time**: 45 mins

---

## 1. Indexing Algorithms

Brute force search ($O(N)$) is too slow for 1 Billion vectors. We need Approximate Nearest Neighbor (ANN).

### 1.1 HNSW (Hierarchical Navigable Small World)
*   **Structure**: A multi-layer graph. Top layers have long links (highways). Bottom layers have short links (local roads).
*   **Search**: Start at top, greedy traverse to finding closest node, drop down layer, repeat.
*   **Pros**: Fastest recall/latency trade-off.
*   **Cons**: High memory usage (stores graph edges).

### 1.2 IVF-PQ (Inverted File with Product Quantization)
*   **IVF**: Cluster vectors into $K$ lists (Voronoi cells). Only search the closest lists.
*   **PQ**: Compress vectors (e.g., 1024 float32 -> 64 bytes).
*   **Pros**: Low memory. Good for billion-scale.

---

## 2. The Ecosystem

### 2.1 Specialized Vector DBs
*   **Pinecone**: Managed, proprietary. Good scaling.
*   **Milvus / Qdrant**: Open source, Go/Rust based. High performance.
*   **Weaviate**: Hybrid search focus.

### 2.2 General DBs with Vector Support
*   **pgvector (PostgreSQL)**: Good enough for < 10M vectors. Keeps data in one place.
*   **Elasticsearch**: Strong hybrid search.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Pre-Filtering vs. Post-Filtering
**Scenario**: Query "Red Shoes". Filter `color='red'`.
*   **Post-Filtering**: Search Top 100 vectors. Filter for red. If only 2 are red, you return 2 results. (Bad Recall).
*   **Pre-Filtering**: Filter for red first. Then search vectors.
    *   *Problem*: If HNSW graph is built on all data, the "red" subset might be disconnected in the graph.
    *   *Solution*: **Filtered HNSW** (modern DBs handle this) or brute force if filter is very selective.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: How does Product Quantization (PQ) work?**
> **Answer**:
> 1.  Split a 128-dim vector into 8 sub-vectors of 16-dims.
> 2.  Run K-Means on each sub-space to get 256 centroids.
> 3.  Replace each sub-vector with the ID of the closest centroid (1 byte).
> 4.  Total size: 8 bytes. Compression: 64x.

**Q2: When should you use pgvector over Pinecone?**
> **Answer**:
> *   **pgvector**: When you already use Postgres, have < 10M vectors, and want transactional consistency (ACID) between your metadata and vectors.
> *   **Pinecone**: When you need massive scale (100M+), low latency (<10ms), and don't want to manage infrastructure.

**Q3: What is the "Curse of Dimensionality" in Vector Search?**
> **Answer**: As dimensions increase, the distance between the nearest and farthest point becomes negligible. All points become equidistant. However, real-world data lies on a lower-dimensional manifold, so ANN search still works effectively.

---

## 5. Further Reading
- [Faiss: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [HNSW Algorithm Explained](https://www.pinecone.io/learn/series/faiss/hnsw/)
