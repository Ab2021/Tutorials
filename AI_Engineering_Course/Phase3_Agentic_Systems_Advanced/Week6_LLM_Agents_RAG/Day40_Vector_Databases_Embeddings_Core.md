# Day 40: Vector Databases & Embeddings
## Core Concepts & Theory

### The Vector Database Revolution

**Traditional Databases:**
- Store structured data (rows, columns).
- Query using SQL (exact matches, ranges).
- **Limitation:** Cannot search by semantic similarity.

**Vector Databases:**
- Store high-dimensional vectors (embeddings).
- Query using similarity metrics (cosine, dot product).
- **Use Case:** Semantic search, RAG, recommendation systems.

### 1. Embedding Fundamentals

**What is an Embedding?**
- A dense vector representation of data (text, image, audio).
- **Dimension:** Typically 384-4096 for text.
- **Property:** Similar items have similar embeddings.

**Example:**
```
"cat" → [0.2, 0.8, -0.3, ..., 0.5]  (1536 dimensions)
"dog" → [0.3, 0.7, -0.2, ..., 0.4]  (similar to "cat")
"car" → [-0.5, 0.1, 0.9, ..., -0.2] (different from "cat")
```

**Embedding Models:**
- **OpenAI:** text-embedding-3-small (1536d), text-embedding-3-large (3072d)
- **Cohere:** embed-english-v3.0 (1024d), embed-multilingual-v3.0 (1024d)
- **Sentence-BERT:** all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
- **BGE:** bge-large-en-v1.5 (1024d)

### 2. Vector Database Architecture

**Core Components:**

**Storage Layer:**
- Store vectors efficiently (compressed, quantized).
- **Format:** Float32 (4 bytes/dim), Int8 (1 byte/dim), Binary (1 bit/dim).

**Index Layer:**
- Build index for fast similarity search.
- **Algorithms:** HNSW, IVF, LSH, Product Quantization.

**Query Layer:**
- Process similarity queries.
- **Operations:** KNN (K-Nearest Neighbors), Range Search, Hybrid Search.

**Metadata Layer:**
- Store metadata alongside vectors.
- **Filtering:** Filter by date, category, author before similarity search.

### 3. Indexing Algorithms

**HNSW (Hierarchical Navigable Small World):**
- **Structure:** Multi-layer graph.
- **Search:** Navigate from top layer to bottom.
- **Complexity:** O(log N) search time.
- **Trade-off:** High memory usage, excellent recall.

**IVF (Inverted File Index):**
- **Structure:** Cluster vectors, create inverted index.
- **Search:** Find nearest cluster, search within cluster.
- **Complexity:** O(√N) search time.
- **Trade-off:** Lower memory, good for large datasets.

**Product Quantization (PQ):**
- **Concept:** Compress vectors by quantizing sub-vectors.
- **Compression:** 32x reduction (1024d float32 → 32 bytes).
- **Trade-off:** Lossy compression, slight accuracy drop.

### 4. Similarity Metrics

**Cosine Similarity:**
$$\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}$$
- **Range:** [-1, 1]. 1 = identical, 0 = orthogonal, -1 = opposite.
- **Use Case:** Text embeddings (most common).

**Euclidean Distance (L2):**
$$d(A, B) = \sqrt{\sum (a_i - b_i)^2}$$
- **Range:** [0, ∞]. 0 = identical, larger = more different.
- **Use Case:** Image embeddings.

**Dot Product:**
$$\text{dot}(A, B) = \sum a_i b_i$$
- **Range:** (-∞, ∞). Higher = more similar.
- **Use Case:** When vectors are normalized.

### 5. Vector Database Comparison

| Database | Type | Index | Filtering | Cloud | Open Source |
|:---------|:-----|:------|:----------|:------|:------------|
| **Pinecone** | Managed | HNSW | Yes | Yes | No |
| **Weaviate** | Self-hosted | HNSW | Yes | Optional | Yes |
| **Qdrant** | Self-hosted | HNSW | Yes | Optional | Yes |
| **Milvus** | Self-hosted | IVF, HNSW | Yes | Optional | Yes |
| **Chroma** | Embedded | HNSW | Yes | No | Yes |
| **FAISS** | Library | IVF, HNSW, PQ | No | No | Yes |
| **pgvector** | Postgres Extension | IVF | Yes | Optional | Yes |

### 6. Metadata Filtering

**Concept:** Filter by metadata before similarity search.

**Example:**
```python
# Find similar documents from last 30 days, department="Engineering"
results = vector_db.search(
    query_embedding,
    filter={
        "date": {"$gte": "2024-01-01"},
        "department": "Engineering"
    },
    top_k=10
)
```

**Benefits:**
- Reduces search space.
- Ensures results meet criteria.
- Faster than post-filtering.

### 7. Hybrid Search

**Concept:** Combine vector search (semantic) with keyword search (exact match).

**Process:**
1. **Vector Search:** Find top 20 by similarity.
2. **Keyword Search:** Find top 20 by BM25.
3. **Fusion:** Combine using RRF or weighted sum.

**Use Case:** "Find documents about 'neural networks' from 2024"
- Vector search: Captures "deep learning", "transformers" (semantic).
- Keyword search: Ensures "neural networks" appears (exact match).

### 8. Quantization

**Scalar Quantization:**
- Convert float32 → int8.
- **Compression:** 4x reduction.
- **Accuracy:** ~1-2% recall drop.

**Product Quantization:**
- Split vector into sub-vectors, quantize each.
- **Compression:** 8-32x reduction.
- **Accuracy:** ~5-10% recall drop.

**Binary Quantization:**
- Convert to binary (0 or 1).
- **Compression:** 32x reduction.
- **Accuracy:** ~10-20% recall drop.

### 9. Sharding and Replication

**Sharding:**
- Split data across multiple nodes.
- **Benefit:** Handle billions of vectors.
- **Challenge:** Cross-shard queries.

**Replication:**
- Duplicate data across nodes.
- **Benefit:** High availability, fault tolerance.
- **Challenge:** Consistency.

### 10. Real-World Use Cases

**Semantic Search:**
- Search documents by meaning, not keywords.
- **Example:** Notion AI, ChatGPT with browsing.

**Recommendation Systems:**
- Find similar products, content.
- **Example:** Spotify (similar songs), Netflix (similar shows).

**Anomaly Detection:**
- Find outliers in high-dimensional data.
- **Example:** Fraud detection, network intrusion.

**Deduplication:**
- Find near-duplicate documents.
- **Example:** Remove duplicate support tickets.

### Summary

**Vector DB Benefits:**
- Semantic search (meaning, not keywords).
- Scalable (billions of vectors).
- Fast (millisecond queries).
- Flexible (metadata filtering, hybrid search).

**Vector DB Challenges:**
- Storage cost (high-dimensional vectors).
- Index build time (hours for billions of vectors).
- Accuracy vs speed trade-off.
- Cold start (no data initially).

### Next Steps
In the Deep Dive, we will implement a production vector database with Pinecone, Qdrant, and pgvector, including quantization and hybrid search.
