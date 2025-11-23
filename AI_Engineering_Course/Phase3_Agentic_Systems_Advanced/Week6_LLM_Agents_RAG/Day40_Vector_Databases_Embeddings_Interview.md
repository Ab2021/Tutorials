# Day 40: Vector Databases & Embeddings
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is a vector database and how does it differ from a traditional database?

**Answer:**
- **Traditional DB:** Stores structured data (rows/columns). Queries use exact matches (SQL: WHERE name = 'John').
- **Vector DB:** Stores high-dimensional vectors (embeddings). Queries use similarity (find vectors similar to query vector).
- **Use Case:** Traditional DB for transactions, Vector DB for semantic search, RAG, recommendations.
- **Example:** "Find documents similar to 'machine learning'" requires vector DB, not SQL.

#### Q2: Explain HNSW and why it's popular for vector search.

**Answer:**
- **HNSW:** Hierarchical Navigable Small World. Multi-layer graph structure.
- **Search:** Start at top layer (sparse), navigate to bottom layer (dense). O(log N) complexity.
- **Benefits:** Fast search, high recall (>95%), works well for millions of vectors.
- **Trade-off:** High memory usage (stores full graph), slower inserts.
- **Used by:** Pinecone, Weaviate, Qdrant.

#### Q3: What is Product Quantization and when should you use it?

**Answer:**
- **PQ:** Compression technique. Split vector into sub-vectors, quantize each independently.
- **Compression:** 8-32x reduction (1024d float32 → 32-128 bytes).
- **Accuracy:** ~5-10% recall drop.
- **When to Use:** Billions of vectors, limited memory, can tolerate slight accuracy loss.
- **Example:** Storing 1B vectors in 32 GB instead of 4 TB.

#### Q4: What is hybrid search and why is it useful?

**Answer:**
- **Hybrid:** Combine vector search (semantic) + keyword search (BM25).
- **Why:** Vector search misses exact keyword matches. BM25 misses semantic similarity.
- **Example:** Query "neural networks 2024"
  - Vector: Finds "deep learning", "transformers" (semantic).
  - BM25: Ensures "neural networks" and "2024" appear (exact).
- **Fusion:** Use RRF or weighted sum to combine rankings.

#### Q5: How do you choose between Pinecone, Qdrant, and pgvector?

**Answer:**
- **Pinecone:** Fully managed, easiest to use, expensive. Use for quick prototypes or if you don't want to manage infrastructure.
- **Qdrant:** Self-hosted, fast, feature-rich. Use for production with control over infrastructure.
- **pgvector:** Postgres extension, simple, integrates with existing Postgres. Use if you already use Postgres and have <10M vectors.
- **Decision:** Pinecone (ease) > Qdrant (performance) > pgvector (simplicity).

---

### Production Challenges

#### Challenge 1: Index Build Time

**Scenario:** You have 100M vectors. Building HNSW index takes 24 hours.
**Root Cause:** HNSW construction is O(N log N).
**Solution:**
- **Incremental Indexing:** Build index in batches. Add 1M vectors at a time.
- **Parallel Construction:** Use multiple workers to build index in parallel.
- **Pre-built Index:** If data is static, build index once offline, deploy.
- **Alternative Index:** Use IVF (faster build, slightly lower recall).

#### Challenge 2: Memory Overflow

**Scenario:** 10M vectors * 1536 dimensions * 4 bytes = 61 GB. Your server has 32 GB RAM.
**Solution:**
- **Quantization:** Use int8 (15 GB) or Product Quantization (2 GB).
- **Disk-Based Index:** Use Milvus or Qdrant with disk storage (slower but fits).
- **Sharding:** Split across multiple nodes (5M vectors per node).
- **Cloud:** Use managed service (Pinecone auto-scales).

#### Challenge 3: Cold Start Problem

**Scenario:** New vector DB with no data. Search quality is poor.
**Root Cause:** No vectors to retrieve.
**Solution:**
- **Seed Data:** Pre-populate with common documents (FAQs, documentation).
- **Fallback:** If no results, fall back to keyword search or LLM knowledge.
- **Incremental:** As users add data, quality improves.

#### Challenge 4: Metadata Filtering Performance

**Scenario:** You filter by `department="Engineering"`. Only 1% of vectors match. Search is slow.
**Root Cause:** Vector DB searches all vectors, then filters (post-filtering).
**Solution:**
- **Pre-Filtering:** Use a DB that supports pre-filtering (Qdrant, Weaviate).
- **Separate Indexes:** Create one index per department.
- **Hybrid Approach:** Use traditional DB for filtering, then vector search on subset.

#### Challenge 5: Embedding Drift

**Scenario:** You update your embedding model (OpenAI v2 → v3). Old embeddings are incompatible.
**Root Cause:** Different embedding spaces.
**Solution:**
- **Re-Embed Everything:** Generate new embeddings for all documents (expensive).
- **Versioning:** Maintain multiple indexes (v2, v3). Gradually migrate.
- **Backward Compatibility:** Use models with backward compatibility (rare).

### Summary Checklist for Production
- [ ] **Index:** Use **HNSW** for <10M vectors, **IVF** for >10M.
- [ ] **Quantization:** Use **int8** for 4x compression, **PQ** for 32x.
- [ ] **Hybrid Search:** Combine **vector + BM25** for best results.
- [ ] **Metadata:** Use **pre-filtering** if available.
- [ ] **Monitoring:** Track **recall**, **latency**, and **memory usage**.
- [ ] **Backup:** **Snapshot** index regularly for disaster recovery.
