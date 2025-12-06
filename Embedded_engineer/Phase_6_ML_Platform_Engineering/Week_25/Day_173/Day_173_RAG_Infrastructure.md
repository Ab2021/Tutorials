# Day 173: The Long Term Memory: RAG Infrastructure
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** LLMs hallucinate. They don't know your private data. **Retrieval Augmented Generation (RAG)** fetches relevant facts from a Vector Database and feeds them to the LLM. Scaling this requires **HNSW** and **Reranking**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** HNSW (Hierarchical Navigable Small World) indexing for billion-scale search.
2.  **Deploy** a Milvus or Qdrant Vector Database Cluster.
3.  **Implement** Hybrid Search (Sparse BM25 + Dense Vectors) for better recall.
4.  **Optimize** precision using a Cross-Encoder Reranker.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install pymilvus sentence-transformers rank_bm25`.
- Docker Compose (for Milvus).

---

## 📖 Theoretical Foundation

### 1. Vector Search (ANN)
*   **Brute Force (KNN):** Calculate distance to ALL 1B vectors. Exact but slow (seconds).
*   **IVF (Inverted File):** Cluster vectors. Only search closest clusters. Faster.
*   **HNSW (Graph):** Build a multi-layer graph. Zoom in from top layer to bottom layer. Fast (ms) and high recall.

### 2. The Lost in the Middle Problem
Retrieving 100 documents might confuse the LLM.
Most LLMs pay attention to the *beginning* and *end* of the context.
**Reranking:**
1.  Retrieve 100 docs (Bi-Encoder, Fast).
2.  Score them against query with Heavy Model (Cross-Encoder, Slow).
3.  Select Top 5.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Deploy Milvus (Docker)

Scale-out vector database.

#### 📁 `docker-compose.yml`
```yaml
version: '3.5'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.0
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  milvus:
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
    depends_on:
      - etcd
      - minio
```

### 👨‍💻 Core Implementation: HNSW Indexing

#### 📁 `src/index_milvus.py`
```python
from pymilvus import (
    connections, FieldSchema, CollectionSchema, 
    DataType, Collection, utility
)
from sentence_transformers import SentenceTransformer

# 1. Connect
connections.connect("default", host="localhost", port="19530")

# 2. Schema
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, "Knowledge Base")
collection = Collection("rag_docs", schema)

# 3. Insert
model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["Kubernetes is a container orchestrator.", "Python is a language."]
vectors = model.encode(docs)

collection.insert([
    docs,
    vectors
])

# 4. Build Index (HNSW)
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}
collection.create_index("embeddings", index_params)
collection.load()
```

### 👨‍💻 Core Implementation: Reranking Pipeline

#### 📁 `src/rag_pipeline.py`
```python
from sentence_transformers import CrossEncoder

# 1. Bi-Encoder (Fast Retrieval)
def retrieve(query, k=50):
    query_vec = model.encode([query])
    res = collection.search(
        query_vec, "embeddings", 
        param={"metric_type": "L2", "params": {"ef": 10}}, 
        limit=k, output_fields=["text"]
    )
    return [hit.entity.get("text") for hit in res[0]]

# 2. Cross-Encoder (Accurate Re-ranking)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rag_search(query):
    # Step 1: Get Candidates
    candidates = retrieve(query, k=50) # Improve Recall
    
    # Step 2: Score Pairs (Query, Doc)
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Step 3: Sort
    ranked_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked_docs[:5] # Return Top 5

# Usage
# context = rag_search("How do I deploy pods?")
# full_prompt = f"Context: {context}\n\nQuestion: ..."
```

---

## 🔬 Lab Exercise: "The Keyword Trap"

### Task
Compare Dense vs Hybrid Search.
1.  **Query:** "IT Error 500".
2.  **Dense Retrieval:** Might return "Server Crash" (Semantic match).
3.  **Keyword Retrieval:** Returns "Log entry: Error 500" (Exact match).
4.  **Scenario:** If the user searches for a specific Error Code (e.g., "0x8004"), Dense Vector Search often fails because numbers have poor semantic embedding.
5.  **Fix:** Use Hybrid Search (Splade or BM25 + Vector). Combine scores: `WeightedScore = 0.7 * Vector + 0.3 * BM25`.

---

## 📖 Advanced Theory: Chunking Strategy
How to split PDF?
*   **Fixed Size:** 512 chars. (Bad: Cuts sentences).
*   **Recursive:** Split by Paragraph, then Sentence. (Better).
*   **Semantic Chunking:** Calculate embedding of sentence $i$ and $i+1$. If distance > threshold, start new chunk. (Best). keeps coherent topics together.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Recall is King:** If the vector DB doesn't find the document, the LLM cannot answer. Optimization priority: Indexing > Reranking > LLM Prompting.
2.  **Latency:** Cross-Encoders are slow (e.g., 200ms). Only rerank top 50 docs, not top 1000.
3.  **Metadata Filtering:** "Show me docs about 'deployment' authored by 'Alice' in '2023'". Apply Scalar Filters *before* Vector Search (Pre-filtering) for speed.

### API Summary
```python
collection.search(param={"ef": 64})
reranker.predict(pairs)
```

---

**Day 173 Complete** ✅

*Next: Day 174 - Prompt Engineering & Evaluation - DSPy.*
