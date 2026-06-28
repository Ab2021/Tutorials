# VECTOR DATABASES COMPLETE — Qdrant, Weaviate, LanceDB, Pinecone, Elasticsearch
## When to use each, production trade-offs, and interview-ready deep dives

---

## SECTION 1: THE LANDSCAPE — WHAT EACH DB IS

### Quick Selection Guide

| Database | Best For | Hosting | GDPR/On-Prem | Cost | Maturity |
|----------|----------|---------|--------------|------|---------|
| **Qdrant** | Production RAG, SME use cases, full control | Self-hosted or Cloud | ✅ Excellent (Rust, self-hosted) | Free (self-hosted), $25+/mo cloud | High |
| **Weaviate** | Multi-modal, hybrid search, GraphQL API | Self-hosted or Cloud | ✅ Good | Free (self-hosted), usage-based cloud | High |
| **LanceDB** | Embedded, serverless, local deployment | Embedded (no server) | ✅ Perfect (local files) | Free | Medium |
| **Pinecone** | Fully managed, no infra ops | Cloud only | ❌ Data in US/EU | $70+/mo | High |
| **Elasticsearch** | Full-text + vector, existing ES stack | Self-hosted or Elastic Cloud | ✅ Good | Free (self-hosted) | Very High |
| **Chroma** | Development, small projects | Embedded or Client-Server | ✅ Good | Free | Medium |
| **FAISS** | Libraries, not a DB, batch offline | In-process | ✅ Perfect | Free | Very High |

---

## SECTION 2: QDRANT — THE RECOMMENDED CHOICE FOR SME AI

### Why Qdrant for This Role

> "For Italian SME clients, Qdrant is my default vector database. It's written in Rust — extremely fast and memory-efficient. It's fully self-hosted, so client data never leaves their infrastructure (GDPR compliance). It has native support for hybrid search (dense + sparse vectors), rich metadata filtering, and multi-tenancy via payload-based filtering or separate collections. The REST API and Python client are clean and production-ready."

### Architecture

```
Qdrant Components:
- Collection: a named vector store (like a table in SQL)
- Point: a single vector + payload (metadata) entry
- Payload: any JSON metadata attached to a vector
- Segment: internal storage unit (Qdrant manages automatically)
- Shard: for distributed deployments (horizontal scaling)
```

### Production Setup with Docker

```yaml
# docker-compose.yml for SME deployment
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.9.0  # Pin to specific version!
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC (faster for batch operations)
    volumes:
      - ./qdrant_storage:/qdrant/storage  # Persist data outside container
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}  # Auth for production
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### Collection Creation and Configuration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    HnswConfigDiff, OptimizersConfigDiff, QuantizationConfig,
    ScalarQuantizationConfig, ScalarType
)

client = QdrantClient(url="http://localhost:6333", api_key=os.getenv("QDRANT_API_KEY"))

def create_collection(collection_name: str, vector_size: int = 1536):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            # Dense vector for semantic search
            "dense": VectorParams(
                size=vector_size,              # 1536 for text-embedding-3-small
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=16,                      # Number of edges per node (higher = better quality, more memory)
                    ef_construct=100,          # Build time quality (higher = slower build, better index)
                    full_scan_threshold=10000  # Full scan for small collections (faster than HNSW for tiny sets)
                )
            )
        },
        sparse_vectors_config={
            # Sparse vector for BM25-style keyword search
            "sparse": SparseVectorParams()
        },
        # Quantization: reduce memory usage by 4x with ~1% quality loss
        quantization_config=QuantizationConfig(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,  # Calibrate on 99th percentile of values
                always_ram=True  # Keep quantized vectors in RAM for speed
            )
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000  # Only build HNSW index when >20K vectors
        )
    )

# Create the collection
create_collection("sme_documents")
```

### Inserting Points

```python
from qdrant_client.models import PointStruct, SparseVector

def upsert_document_chunks(chunks: list[dict], collection_name: str):
    """Upsert document chunks into Qdrant"""
    
    points = []
    for i, chunk in enumerate(chunks):
        # Get dense embedding
        dense_embedding = embed_model.encode(chunk["text"])
        
        # Get sparse embedding (BM25)
        sparse_indices, sparse_values = bm25_encoder.encode(chunk["text"])
        
        points.append(PointStruct(
            id=chunk["chunk_id"],  # Must be int or UUID
            vector={
                "dense": dense_embedding.tolist(),
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
            },
            payload={
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "page_number": chunk["page_number"],
                "client_id": chunk["client_id"],      # Tenant isolation
                "doc_type": chunk["doc_type"],        # "invoice", "contract", "email"
                "created_at": chunk["created_at"],
                "language": chunk["language"],         # "it", "en"
                "chunk_index": i                       # Position in document
            }
        ))
    
    # Batch upsert (much faster than one-by-one)
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True  # Wait for indexing to complete before returning
    )
```

### Searching — Dense, Sparse, and Hybrid

```python
from qdrant_client.models import (
    SearchRequest, Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion
)

def dense_search(query: str, client_id: str, doc_type: str = None, top_k: int = 10):
    """Pure semantic search"""
    query_embedding = embed_model.encode(query)
    
    # Build filter for tenant isolation + optional doc type
    filter_conditions = [FieldCondition(key="client_id", match=MatchValue(value=client_id))]
    if doc_type:
        filter_conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    
    results = client.search(
        collection_name="sme_documents",
        query_vector=("dense", query_embedding.tolist()),
        query_filter=Filter(must=filter_conditions),
        limit=top_k,
        with_payload=True,       # Return metadata
        with_vectors=False,      # Don't return vectors (saves bandwidth)
        score_threshold=0.70     # Minimum similarity score
    )
    return results

def hybrid_search(query: str, client_id: str, top_k: int = 10, alpha: float = 0.5):
    """Hybrid search: semantic + keyword (Reciprocal Rank Fusion)"""
    dense_vector = embed_model.encode(query).tolist()
    sparse_indices, sparse_values = bm25_encoder.encode(query)
    
    results = client.query_points(
        collection_name="sme_documents",
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=20,
                filter=Filter(must=[FieldCondition(key="client_id", match=MatchValue(value=client_id))])
            ),
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=20,
                filter=Filter(must=[FieldCondition(key="client_id", match=MatchValue(value=client_id))])
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
        limit=top_k,
        with_payload=True
    )
    return results
```

### HNSW — The Index Algorithm Behind Qdrant

**What HNSW is (interview question):**
> "HNSW is Hierarchical Navigable Small World — a graph-based approximate nearest neighbor (ANN) index. It creates a multi-layer graph where higher layers have fewer, widely-connected nodes for fast coarse navigation, and lower layers have all nodes with fine-grained connections. Search starts at the top layer and greedily traverses to the query's nearest neighbors at each layer. This gives O(log N) search complexity vs O(N) for brute force."

**Key parameters:**
- `m` (16 default): edges per node. More edges → better quality, more memory, slower build
- `ef_construct` (100): build-time search depth. Higher → better index quality, slower indexing
- `ef` (query-time): search depth during query. Higher → more accurate, slower

---

## SECTION 3: WEAVIATE — MULTI-MODAL AND GRAPHQL

### When to Choose Weaviate

```
Choose Weaviate when:
✅ Multi-modal data (text + images + video)
✅ Need GraphQL API for complex queries
✅ Built-in transformers integration (no separate embedding step)
✅ Concept-level search (not just similarity)

Choose Qdrant when:
✅ Pure vector similarity performance
✅ On-premise with minimal resource requirements
✅ More complex filtering needed
✅ Lower memory footprint
```

### Weaviate Schema and Data Classes

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

client = weaviate.connect_to_local()  # For self-hosted

# Define schema (data class)
client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"  # Auto-vectorizes on insert!
    ),
    generative_config=Configure.Generative.openai(model="gpt-4o"),  # RAG built-in
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="client_id", data_type=DataType.TEXT),
        Property(name="doc_type", data_type=DataType.TEXT),
        Property(name="created_at", data_type=DataType.DATE),
    ]
)

# Weaviate's built-in RAG:
documents = client.collections.get("Document")
response = documents.generate.near_text(
    query="payment terms",
    single_prompt="Summarize the payment terms from this context: {content}",
    filters=weaviate.classes.query.Filter.by_property("client_id").equal("client_123"),
    limit=5
)
```

---

## SECTION 4: LANCEDB — EMBEDDED, SERVERLESS (PERFECT FOR SME ON-PREM)

### Why LanceDB for Italian SME On-Premise

> "LanceDB is an embedded vector database — it runs as a library inside your application, not as a separate server. Data is stored as Apache Lance files on local disk or S3-compatible storage. For Italian SME clients with strict data residency requirements, this is perfect: no database server to manage, data stays on their file system, and it integrates directly into a Python application."

### LanceDB Setup and Usage

```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
import numpy as np

# Connect to local storage (no server needed!)
db = lancedb.connect("/path/to/client_data/lancedb")  # Or S3: "s3://bucket/lancedb"

# Define schema using Pydantic
class DocumentChunk(LanceModel):
    id: str
    text: str
    vector: Vector(1536)  # Embedding dimension
    doc_id: str
    doc_name: str
    page_number: int
    client_id: str
    doc_type: str
    created_at: str

# Create table (or open existing)
table = db.create_table("documents", schema=DocumentChunk, exist_ok=True)

# Insert data
def index_chunks(chunks: list[DocumentChunk]):
    table.add(chunks)  # Batch insert
    table.create_fts_index("text", replace=True)  # Full-text search index

# Search
def search(query: str, client_id: str, top_k: int = 5) -> list:
    query_embedding = embed_model.encode(query)
    
    results = (
        table.search(query_embedding, vector_column_name="vector")
        .where(f"client_id = '{client_id}'")   # Metadata filter
        .limit(top_k)
        .to_pydantic(DocumentChunk)
    )
    return results

# Hybrid search (vector + full-text)
def hybrid_search(query: str, client_id: str, top_k: int = 5):
    query_embedding = embed_model.encode(query)
    
    results = (
        table.search(query_embedding)
        .where(f"client_id = '{client_id}'")
        .rerank(reranker=lancedb.rerankers.CrossEncoderReranker())  # Built-in reranking!
        .limit(top_k)
        .to_pandas()
    )
    return results
```

### When LanceDB Over Qdrant

```
LanceDB wins when:
- No server management (embedded library)
- Data must never leave a specific machine
- Air-gapped environments (manufacturing plant)
- Simpler deployment: just files, no Docker/K8s
- Smaller scale (<10M vectors)

Qdrant wins when:
- Scale (100M+ vectors)
- Multi-tenant with strict isolation
- High concurrent query load
- Need REST API for external services
- Distributed deployment
```

---

## SECTION 5: ELASTICSEARCH / OPENSEARCH — FULL-TEXT + VECTOR

### Why Elasticsearch Matters

Many Italian SMEs already have Elasticsearch for log management or product search. Adding vector search to existing ES = low infrastructure footprint.

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("user", "password"),
    verify_certs=True,
    ca_certs="/path/to/ca.crt"
)

# Create index with both text and vector fields
es.indices.create(
    index="sme_documents",
    body={
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "italian"},  # Italian language analyzer!
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                },
                "client_id": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "created_at": {"type": "date"}
            }
        }
    }
)

# Hybrid search: BM25 text + vector in one query
def es_hybrid_search(query: str, client_id: str, top_k: int = 10):
    query_embedding = embed_model.encode(query).tolist()
    
    response = es.search(
        index="sme_documents",
        body={
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": {"query": query, "boost": 0.3}}}  # BM25 component
                    ],
                    "filter": [
                        {"term": {"client_id": client_id}}  # Tenant isolation
                    ]
                }
            },
            "knn": {
                "field": "dense_vector",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": 100,
                "boost": 0.7  # Semantic component weight
            },
            "size": top_k,
            "_source": {"includes": ["text", "doc_name", "page_number", "client_id"]}
        }
    )
    return response["hits"]["hits"]
```

**Italian language analyzer — CRITICAL detail for this role:**
```json
{
  "analyzer": {
    "italian": {
      "type": "custom",
      "tokenizer": "standard",
      "filter": ["lowercase", "italian_stop", "italian_stemmer"]
    }
  },
  "filter": {
    "italian_stop": {
      "type": "stop",
      "stopwords": "_italian_"
    },
    "italian_stemmer": {
      "type": "stemmer",
      "language": "italian"
    }
  }
}
```

---

## SECTION 6: VECTOR DB COMPARISON TABLE — INTERVIEW READY

| Dimension | Qdrant | Weaviate | LanceDB | Pinecone | Elasticsearch |
|-----------|--------|----------|---------|----------|---------------|
| **Hosting** | Self/Cloud | Self/Cloud | Embedded | Cloud only | Self/Cloud |
| **GDPR / On-prem** | ✅ Perfect | ✅ Good | ✅ Perfect | ❌ Cloud only | ✅ Good |
| **Hybrid search** | ✅ Native | ✅ Native | ✅ With FTS | ✅ Native | ✅ Excellent |
| **Multi-tenancy** | ✅ Collections/Filter | ✅ Multi-tenancy built-in | ✅ Filter-based | ✅ Namespaces | ✅ Indices/Filter |
| **Scale** | Millions | Millions | <10M best | Billions | Billions |
| **Memory usage** | Low (Rust) | Medium | Low | N/A (managed) | High |
| **Query speed** | Very fast | Fast | Fast | Fast | Fast (for existing users) |
| **Italian language** | Filter-based | ✅ Analyzers | Filter-based | Filter-based | ✅ Italian analyzer built-in |
| **Setup complexity** | Docker (simple) | Docker (medium) | Library import | None | Docker (medium) |
| **Operational complexity** | Low | Medium | None | None | High |
| **Cost (self-hosted)** | Free | Free | Free | N/A | Free (but RAM heavy) |

---

## SECTION 7: INTERVIEW QUESTIONS AND ANSWERS

**Q: "What vector database would you choose for an Italian manufacturing client who stores their data on-premise?"**
> "LanceDB for small scale (<1M documents), Qdrant for medium-large scale. Both self-host with no server management overhead for LanceDB (embedded) or Docker for Qdrant. Both keep data on Italian infrastructure, satisfying GDPR data residency requirements. I'd use multilingual-e5-large as the embedding model locally — no API calls to OpenAI, fully air-gapped operation."

**Q: "Explain HNSW indexing"**
> "HNSW builds a layered graph. Top layers have long-range connections for fast traversal, bottom layer has all vectors with local connections. Search starts at the top, greedily follows closest nodes at each layer down to the bottom. Build complexity is O(N log N), query complexity O(log N). The tradeoff vs flat search: approximate results (recall ~95-99% configurable), 100x faster for large collections."

**Q: "What is the difference between cosine similarity and dot product for vector search?"**
> "Cosine similarity measures the ANGLE between vectors — purely directional, independent of magnitude. Dot product measures both direction AND magnitude. For normalized embeddings (most models output unit vectors), they're identical. For un-normalized vectors, dot product can rank semantically similar but different-magnitude vectors differently. Most embedding models (OpenAI, BERT) output normalized vectors, so cosine and dot product give same results. QDRANT uses cosine by default for text."

**Q: "How would you handle adding new documents without re-indexing everything?"**
> "Both Qdrant and LanceDB support incremental upserts — add new vectors without rebuilding the index. For large collections (HNSW), the index updates in-place. The key operational challenge is tracking which documents are indexed: I maintain a document registry table (SQLite or Postgres) with doc_id, content_hash, indexed_at. On new document upload: compute hash, check registry, upsert only if hash changed or doc is new. For deleted documents: Qdrant delete_points by doc_id, then remove from registry."
