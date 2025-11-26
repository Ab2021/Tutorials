# Day 28: Vector Databases in Production - Powering AI Applications

## Table of Contents
1. [Vector Embeddings Recap](#1-vector-embeddings-recap)
2. [Similarity Search](#2-similarity-search)
3. [Indexing Strategies](#3-indexing-strategies)
4. [Qdrant Production](#4-qdrant-production)
5. [RAG Optimization](#5-rag-optimization)
6. [Hybrid Search](#6-hybrid-search)
7. [Performance Tuning](#7-performance-tuning)
8. [Monitoring & Observability](#8-monitoring--observability)
9. [Production Patterns](#9-production-patterns)
10. [Summary](#10-summary)

---

## 1. Vector Embeddings Recap

### 1.1 What are Embeddings?

**Text → Vector representation**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "I love machine learning"
embedding = model.encode(text)

print(embedding.shape)  # (384,)
print(embedding[:5])    # [0.123, -0.456, 0.789, ...]
```

**Similar texts → Similar vectors**:
```python
text1 = "I love machine learning"
text2 = "AI and ML are amazing"
text3 = "I like pizza"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity([emb1], [emb2]))  # 0.85 (very similar)
print(cosine_similarity([emb1], [emb3]))  # 0.23 (not similar)
```

---

## 2. Similarity Search

### 2.1 Distance Metrics

**Cosine Similarity** (most common):
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Range: -1 to 1 (1 = identical)
```

**Euclidean Distance**:
```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Lower = more similar
```

**Dot Product**:
```python
def dot_product(a, b):
    return np.dot(a, b)

# Higher = more similar (for normalized vectors)
```

### 2.2 Naive Search (Brute Force)

```python
# Search through all vectors
def search(query_embedding, documents, top_k=5):
    similarities = []
    
    for doc in documents:
        sim = cosine_similarity(query_embedding, doc['embedding'])
        similarities.append((sim, doc))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    return similarities[:top_k]

# Problem: O(n) - slow for millions of documents!
```

---

## 3. Indexing Strategies

### 3.1 HNSW (Hierarchical Navigable Small World)

**Concept**: Graph-based index for fast approximate search.

```
Layer 2:  A ←→ B
          ↓     ↓
Layer 1:  A ←→ C ←→ B ←→ D
          ↓     ↓     ↓     ↓
Layer 0:  A ←→ C ←→ E ←→ B ←→ F ←→ D

Search: Start at top layer, navigate to closest, go down
```

**Qdrant HNSW**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    ),
    hnsw_config=HnswConfigDiff(
        m=16,  # Number of connections per node
        ef_construct=100  # Construction search depth
    )
)
```

**Trade-offs**:
- `m` higher → more accurate, more memory
- `ef_construct` higher → slower indexing, more accurate

### 3.2 IVF (Inverted File Index)

**Concept**: Cluster vectors, search only relevant clusters.

```
Cluster 1: [doc1, doc5, doc9] (sports)
Cluster 2: [doc2, doc6, doc10] (tech)
Cluster 3: [doc3, doc7, doc11] (cooking)

Query: "football"
→ Search only Cluster 1 (skip 2 & 3)
```

**Faiss IVF**:
```python
import faiss

# 384-dimensional vectors, 100 clusters
index = faiss.IndexIVFFlat(
    faiss.IndexFlatL2(384),  # Quantizer
    384,  # Dimension
    100   # Number of clusters
)

# Train on sample data
index.train(sample_vectors)

# Add vectors
index.add(vectors)

# Search (nprobe = clusters to search)
index.nprobe = 10
distances, indices = index.search(query_vector, k=5)
```

### 3.3 Product Quantization (PQ)

**Concept**: Compress vectors to save memory.

```python
# 384D → 16 sub-vectors of 24D
# Each sub-vector quantized to 8 bits
# Compression: 384 * 4 bytes → 16 bytes (96x smaller!)

import faiss

index = faiss.IndexPQ(
    384,  # Dimension
    16,   # Sub-vectors
    8     # Bits per sub-vector
)
```

**Trade-off**: ~96x memory savings, ~5% accuracy loss.

---

## 4. Qdrant Production

### 4.1 Cluster Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant-1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__CLUSTER__NODE_ID: "1"
    volumes:
      - ./qdrant-data-1:/qdrant/storage
  
  qdrant-2:
    image: qdrant/qdrant:latest
    ports:
      - "6334:6333"
    environment:
      QDRANT__CLUSTER__ENABLED: "true"
      QDRANT__CLUSTER__NODE_ID: "2"
    volumes:
      - ./qdrant-data-2:/qdrant/storage
```

### 4.2 Sharding

```python
client.create_collection(
    collection_name="large_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    shard_number=4  # Split across 4 shards
)
```

### 4.3 Replication

```python
client.create_collection(
    collection_name="critical_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    replication_factor=3  # 3 copies for high availability
)
```

### 4.4 Backups

```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/documents/snapshots

# Download snapshot
curl http://localhost:6333/collections/documents/snapshots/snapshot-2024-01-01.snapshot \
  --output backup.snapshot

# Restore
curl -X PUT http://localhost:6333/collections/documents/snapshots/upload \
  --data-binary @backup.snapshot
```

---

## 5. RAG Optimization

### 5.1 Chunking Strategies

**Fixed-size chunks**:
```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Overlap to preserve context
    
    return chunks

# Problem: May split mid-sentence
```

**Semantic chunking**:
```python
def semantic_chunk(text, model, threshold=0.7):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    
    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        
        if i < len(sentences) - 1:
            # Check similarity between current and next
            emb1 = model.encode(' '.join(current_chunk))
            emb2 = model.encode(sentences[i + 1])
            
            if cosine_similarity(emb1, emb2) < threshold:
                # Low similarity → topic change → new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 5.2 Metadata Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Search + filter
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="technology")
            ),
            FieldCondition(
                key="published_year",
                range={"gte": 2020}
            )
        ]
    )
)
```

### 5.3 Re-ranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 1. Vector search (fast, approximate)
candidates = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=100  # Get more candidates
)

# 2. Re-rank with cross-encoder (slow, accurate)
pairs = [(query_text, doc['text']) for doc in candidates]
scores = reranker.predict(pairs)

# 3. Sort by re-ranking score
reranked = sorted(zip(scores, candidates), reverse=True, key=lambda x: x[0])
top_10 = reranked[:10]
```

---

## 6. Hybrid Search

### 6.1 Combining Keyword + Vector

```python
from qdrant_client.models import SearchRequest, Prefetch

# Keyword search (BM25)
keyword_results = client.search(
    collection_name="documents",
    query_text="machine learning",  # Full-text search
    limit=50
)

# Vector search
vector_results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=50
)

# Merge results (Reciprocal Rank Fusion)
def rrf_merge(results_list, k=60):
    scores = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Sort by score
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:10]

final_results = rrf_merge([keyword_results, vector_results])
```

### 6.2 Qdrant Hybrid Search

```python
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        should=[  # OR condition
            FieldCondition(key="title", match=MatchText(text="machine learning")),
            FieldCondition(key="content", match=MatchText(text="AI"))
        ]
    ),
    limit=10
)
```

---

## 7. Performance Tuning

### 7.1 Batch Upsert

```python
# Bad: One at a time
for doc in documents:
    client.upsert(collection_name="docs", points=[doc])  # Slow!

# Good: Batch
client.upsert(
    collection_name="docs",
    points=documents  # All at once
)
```

### 7.2 Parallel Search

```python
from concurrent.futures import ThreadPoolExecutor

queries = [query1, query2, query3, ...]

def search_single(query):
    return client.search(collection_name="docs", query_vector=query, limit=10)

# Parallel execution
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(search_single, queries))
```

### 7.3 HNSW Tuning

```python
# Search-time parameter
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    limit=10,
    search_params={
        "hnsw_ef": 128  # Higher = more accurate, slower
        # Default: 128
        # Fast: 64
        # Accurate: 256+
    }
)
```

---

## 8. Monitoring & Observability

### 8.1 Metrics

```python
from prometheus_client import Counter, Histogram

search_duration = Histogram('vector_search_duration_seconds', 'Search latency')
search_count = Counter('vector_search_total', 'Total searches')

@search_duration.time()
def search_vectors(query):
    search_count.inc()
    return client.search(...)
```

### 8.2 Health Checks

```python
@app.get("/health")
def health_check():
    try:
        # Check Qdrant connection
        collections = client.get_collections()
        return {"status": "healthy", "collections": len(collections.collections)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503
```

---

## 9. Production Patterns

### 9.1 Async Indexing

```python
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379')

@celery_app.task
def index_document(doc_id, text):
    # Generate embedding
    embedding = model.encode(text)
    
    # Upsert to Qdrant
    client.upsert(
        collection_name="documents",
        points=[{
            "id": doc_id,
            "vector": embedding.tolist(),
            "payload": {"text": text}
        }]
    )

# Usage
index_document.delay(doc_id, text)  # Async
```

### 9.2 Cache Results

```python
import redis

r = redis.Redis()

def search_with_cache(query_text, query_embedding):
    # Check cache
    cache_key = f"search:{hashlib.md5(query_text.encode()).hexdigest()}"
    cached = r.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Cache miss → search
    results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=10
    )
    
    # Cache for 1 hour
    r.setex(cache_key, 3600, json.dumps(results))
    
    return results
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Embeddings** - Text → 384D vectors
2. ✅ **Similarity Search** - Cosine, euclidean, dot product
3. ✅ **HNSW** - Graph-based fast search
4. ✅ **Qdrant Cluster** - Sharding, replication
5. ✅ **RAG** - Chunking, metadata filtering, re-ranking
6. ✅ **Hybrid Search** - Keyword + vector (RRF)
7. ✅ **Monitoring** - Prometheus metrics

### 10.2 Index Selection

| Index | Speed | Memory | Accuracy | Use Case |
|:------|:------|:-------|:---------|:---------|
| **HNSW** | Fast | High | ~99% | < 10M vectors |
| **IVF** | Medium | Medium | ~95% | 10M-100M |
| **PQ** | Fast | Low | ~90% | 100M+ (memory constrained) |

### 10.3 Tomorrow (Day 29): Database Scaling Patterns

- **Read replicas**: PostgreSQL streaming replication
- **Sharding**: Horizontal partitioning strategies
- **Connection pooling**: PgBouncer, pgpool
- **Query optimization**: EXPLAIN ANALYZE
- **Caching layer**: Redis query cache
- **Database migrations**: Zero-downtime strategies

See you tomorrow! 🚀

---

**File Statistics**: ~950 lines | Vector Databases in Production mastered ✅
