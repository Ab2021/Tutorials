# Day 36 (Part 1): Advanced Vector Databases

> **Phase**: 6 - Deep Dive
> **Topic**: Indexing at Scale
> **Focus**: IVF-PQ, DiskANN, and Filtering
> **Reading Time**: 60 mins

---

## 1. IVF-PQ (Inverted File + Product Quantization)

How to store 1 Billion vectors in RAM?

### 1.1 IVF (Inverted File)
*   Cluster vectors into $K$ centroids (Voronoi cells).
*   At query time, find closest centroid. Search only inside that cell.
*   **Speedup**: Search $N/K$ items instead of $N$.

### 1.2 PQ (Product Quantization)
*   Split 128-dim vector into 8 sub-vectors of 16-dim.
*   Cluster each sub-vector into 256 centroids ($2^8$).
*   Replace sub-vector with Centroid ID (1 byte).
*   **Compression**: 128 floats (512 bytes) -> 8 bytes. 64x compression.

---

## 2. DiskANN (Vamana Graph)

*   **Problem**: HNSW requires RAM. 1B vectors = 1TB RAM ($$$).
*   **Solution**: Store graph on SSD.
*   **Vamana**: A graph structure optimized for fewer disk reads.
*   **Result**: 10x cheaper, slightly higher latency.

---

## 3. Tricky Interview Questions

### Q1: Pre-filtering vs Post-filtering?
> **Answer**:
> *   **Post-filtering**: Search Top K. Then filter by metadata (e.g., "Year > 2020").
>     *   *Risk*: If Top K are all from 2019, result is empty.
> *   **Pre-filtering**: Filter first. Search in subset.
>     *   *Risk*: If subset is small, brute force is fast. If large, index is fragmented.
> *   **Best**: Hybrid / Native Filtering (HNSW with bitmaps).

### Q2: Why does Recall drop with PQ?
> **Answer**:
> *   PQ approximates the distance.
> *   $d(x, y) \approx d(x, q(y))$.
> *   The approximation error might cause the true nearest neighbor to look further away.
> *   **Fix**: Re-ranking. Retrieve Top 100 with PQ. Fetch full vectors from disk. Re-rank exact distance.

### Q3: Milvus vs Pinecone vs FAISS?
> **Answer**:
> *   **FAISS**: Library. You manage the server.
> *   **Milvus**: Open-source Service (K8s). Scalable.
> *   **Pinecone**: Managed SaaS. Easy.

---

## 4. Practical Edge Case: The "Zero Vector"
*   **Problem**: Embedding model outputs all zeros for empty string.
*   **Result**: Cosine similarity is undefined (Divide by zero).
*   **Fix**: Check for norm=0 before indexing.

