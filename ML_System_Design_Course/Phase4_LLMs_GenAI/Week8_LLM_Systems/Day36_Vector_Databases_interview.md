# Day 36: Vector Databases - Interview Questions

> **Topic**: Retrieval Infrastructure
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is a Vector Database? How is it different from SQL?
**Answer:**
*   Stores high-dimensional vectors (embeddings).
*   Optimized for **Similarity Search** (Nearest Neighbor), not exact match.
*   SQL: `WHERE id = 5`. Vector DB: `WHERE distance(vec, query) < threshold`.

### 2. Explain HNSW (Hierarchical Navigable Small World).
**Answer:**
*   Standard indexing algorithm.
*   Multi-layer graph.
*   Top layers: Long jumps (Highways). Bottom layers: Fine-grained search.
*   $O(\log N)$ complexity. Fast and accurate.

### 3. What is IVF (Inverted File Index)?
**Answer:**
*   Cluster vectors into $K$ lists (Voronoi cells).
*   At query time, find closest centroids. Search only within those lists.
*   Speeds up search by pruning search space.

### 4. What is Product Quantization (PQ)?
**Answer:**
*   Compression technique.
*   Split vector into $M$ sub-vectors. Quantize each sub-vector to nearest centroid.
*   Reduces memory usage by 90%+. Allows searching in RAM.

### 5. What is the trade-off between Precision and Recall in Vector Search?
**Answer:**
*   **Exact Search (KNN)**: 100% Recall. Slow ($O(N)$).
*   **Approximate Search (ANN)**: 95-99% Recall. Fast ($O(\log N)$).
*   Trade-off controlled by parameters (`ef_search`, `nprobe`).

### 6. How do you handle Hybrid Search (Text + Vector)?
**Answer:**
*   **Pre-filtering**: Filter by metadata (SQL) -> Vector Search on subset.
*   **Post-filtering**: Vector Search -> Filter results.
*   **Reciprocal Rank Fusion (RRF)**: Combine ranked lists from BM25 and Vector.

### 7. What is "Metadata Filtering"?
**Answer:**
*   "Find similar documents where `year > 2020`".
*   Challenge: If filter is too strict, ANN graph might be disconnected.
*   Solution: Filtered HNSW / Vamana.

### 8. Compare Pinecone, Milvus, and Weaviate.
**Answer:**
*   **Pinecone**: Managed SaaS. Easy. Closed source.
*   **Milvus**: Open source. Scalable (Cloud-native). Complex.
*   **Weaviate**: Open source. Built-in vectorization modules. GraphQL.

### 9. What is DiskANN?
**Answer:**
*   Microsoft algorithm.
*   Stores graph on SSD. Caches compressed vectors in RAM.
*   Allows searching billion-scale datasets on a single machine with small RAM.

### 10. How do you update vectors in a Vector DB?
**Answer:**
*   HNSW is hard to update (graph integrity).
*   Usually: Delete + Insert.
*   LSM Tree approach (Write to memory buffer, merge to disk).

### 11. What distance metrics are used?
**Answer:**
*   **Cosine Similarity**: Normalized dot product. Direction. (Most common).
*   **Euclidean (L2)**: Magnitude matters.
*   **Dot Product**: Unnormalized.

### 12. What is "Dimensionality Curse" in vector search?
**Answer:**
*   In high dimensions, all points become equidistant.
*   Distance metrics lose meaning.
*   Embeddings (768d - 1536d) are usually fine.

### 13. How do you scale a Vector DB?
**Answer:**
*   **Sharding**: Partition vectors by ID.
*   **Replication**: For read throughput.

### 14. What is "Re-ranking"?
**Answer:**
*   Vector DB returns top 100 candidates (fast).
*   Cross-Encoder (slow) re-ranks them for better precision.

### 15. What is a "Collection" vs "Partition"?
**Answer:**
*   **Collection**: Table.
*   **Partition**: Subset of collection.
*   Search can be restricted to a partition.

### 16. How do you handle "Duplicate Vectors"?
**Answer:**
*   Deduplication before insertion.
*   Or allow them (if they represent different chunks/docs).

### 17. What is "Sparse Vector Search" (SPLADE)?
**Answer:**
*   Learned sparse representations.
*   Better than BM25, cheaper than Dense.
*   Supported by some Vector DBs.

### 18. Why is "Pre-filtering" better than "Post-filtering"?
**Answer:**
*   **Post-filtering**: You get 10 results, filter out 8, left with 2. Bad user experience.
*   **Pre-filtering**: You search only valid docs, get 10 valid results.

### 19. What is the "Embedding Model" role?
**Answer:**
*   Vector DB is garbage-in, garbage-out.
*   Quality depends entirely on the Embedding Model (OpenAI, Cohere, E5).

### 20. How do you backup a Vector DB?
**Answer:**
*   Snapshot of the index files.
*   Or export raw vectors + metadata to S3.
