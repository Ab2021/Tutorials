# Day 40: Vector Databases & Embeddings
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. HNSW Algorithm Deep Dive

**Hierarchical Navigable Small World Graph:**

**Construction:**
```python
class HNSW:
    def __init__(self, M=16, ef_construction=200):
        self.M = M  # Max connections per node
        self.ef_construction = ef_construction  # Size of dynamic candidate list
        self.layers = []  # List of layers (graphs)
        self.entry_point = None
    
    def insert(self, vector, id):
        """Insert a vector into the HNSW index."""
        # Determine layer for this vector
        layer = self._select_layer()
        
        # Find nearest neighbors at each layer
        neighbors = self._search_layer(vector, self.entry_point, layer)
        
        # Add connections
        for l in range(layer + 1):
            # Connect to M nearest neighbors
            self._add_connections(id, neighbors[l], l)
        
        # Update entry point if necessary
        if layer > len(self.layers) - 1:
            self.entry_point = id
    
    def _select_layer(self):
        """Select layer using exponential decay."""
        return int(-np.log(np.random.uniform()) * self.ml)
    
    def search(self, query_vector, k=10):
        """Search for k nearest neighbors."""
        # Start from entry point at top layer
        candidates = [self.entry_point]
        
        # Navigate down layers
        for layer in range(len(self.layers) - 1, -1, -1):
            candidates = self._search_layer(query_vector, candidates, layer)
        
        # Return top k
        return candidates[:k]
```

**Time Complexity:**
- **Insert:** O(log N)
- **Search:** O(log N)
- **Space:** O(N * M) where M is max connections

### 2. Product Quantization Implementation

**Concept:** Split vector into sub-vectors, quantize each independently.

```python
import numpy as np
from sklearn.cluster import KMeans

class ProductQuantizer:
    def __init__(self, d=1024, M=8, k=256):
        """
        d: Vector dimension
        M: Number of sub-vectors
        k: Number of centroids per sub-vector
        """
        self.d = d
        self.M = M
        self.k = k
        self.d_sub = d // M  # Sub-vector dimension
        self.codebooks = []  # One codebook per sub-vector
    
    def fit(self, vectors):
        """Train quantizer on vectors."""
        N = len(vectors)
        
        for m in range(self.M):
            # Extract m-th sub-vector from all vectors
            start = m * self.d_sub
            end = (m + 1) * self.d_sub
            sub_vectors = vectors[:, start:end]
            
            # Cluster sub-vectors
            kmeans = KMeans(n_clusters=self.k, random_state=42)
            kmeans.fit(sub_vectors)
            
            # Store codebook (centroids)
            self.codebooks.append(kmeans.cluster_centers_)
    
    def encode(self, vector):
        """Encode vector as sequence of centroid IDs."""
        codes = []
        
        for m in range(self.M):
            start = m * self.d_sub
            end = (m + 1) * self.d_sub
            sub_vector = vector[start:end]
            
            # Find nearest centroid
            distances = np.linalg.norm(self.codebooks[m] - sub_vector, axis=1)
            code = np.argmin(distances)
            codes.append(code)
        
        return np.array(codes, dtype=np.uint8)
    
    def decode(self, codes):
        """Decode centroid IDs back to approximate vector."""
        vector = []
        
        for m in range(self.M):
            centroid = self.codebooks[m][codes[m]]
            vector.extend(centroid)
        
        return np.array(vector)
    
    def compress_database(self, vectors):
        """Compress all vectors."""
        return np.array([self.encode(v) for v in vectors])

# Usage
pq = ProductQuantizer(d=1024, M=8, k=256)
pq.fit(training_vectors)

# Compress
compressed = pq.encode(vector)  # 1024 float32 → 8 uint8 (128x compression!)
```

### 3. Complete Pinecone Integration

```python
import pinecone
import openai

class PineconeRAG:
    def __init__(self, api_key, environment, index_name):
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def upsert_documents(self, documents, metadata_list=None):
        """Add documents to the index."""
        # Generate embeddings
        embeddings = self._embed(documents)
        
        # Prepare vectors
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            metadata = metadata_list[i] if metadata_list else {}
            metadata["text"] = doc  # Store original text
            
            vectors.append({
                "id": f"doc_{i}",
                "values": emb,
                "metadata": metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query, top_k=5, filter=None):
        """Search for similar documents."""
        # Embed query
        query_embedding = self._embed([query])[0]
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Extract documents
        documents = [match["metadata"]["text"] for match in results["matches"]]
        scores = [match["score"] for match in results["matches"]]
        
        return documents, scores
    
    def _embed(self, texts):
        """Generate embeddings using OpenAI."""
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item["embedding"] for item in response["data"]]

# Usage
rag = PineconeRAG(
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="my-rag-index"
)

# Add documents
documents = ["Document 1 text...", "Document 2 text..."]
metadata = [{"date": "2024-01-01", "dept": "Engineering"}, ...]
rag.upsert_documents(documents, metadata)

# Search with filtering
results, scores = rag.search(
    "What is the vacation policy?",
    filter={"dept": {"$eq": "Engineering"}}
)
```

### 4. Qdrant Implementation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantRAG:
    def __init__(self, host="localhost", port=6333, collection_name="documents"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        # Create collection
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except:
            pass  # Collection already exists
    
    def upsert_documents(self, documents, metadata_list=None):
        """Add documents to Qdrant."""
        embeddings = self._embed(documents)
        
        points = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            metadata = metadata_list[i] if metadata_list else {}
            metadata["text"] = doc
            
            points.append(PointStruct(
                id=i,
                vector=emb,
                payload=metadata
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query, top_k=5, filter=None):
        """Search with optional filtering."""
        query_embedding = self._embed([query])[0]
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter
        )
        
        documents = [hit.payload["text"] for hit in results]
        scores = [hit.score for hit in results]
        
        return documents, scores
```

### 5. pgvector (Postgres) Implementation

```python
import psycopg2
from pgvector.psycopg2 import register_vector

class PgVectorRAG:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        register_vector(self.conn)
        
        # Create table
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    embedding vector(1536),
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            self.conn.commit()
    
    def upsert_documents(self, documents, metadata_list=None):
        """Insert documents."""
        embeddings = self._embed(documents)
        
        with self.conn.cursor() as cur:
            for doc, emb, meta in zip(documents, embeddings, metadata_list or [{}] * len(documents)):
                cur.execute(
                    "INSERT INTO documents (text, embedding, metadata) VALUES (%s, %s, %s)",
                    (doc, emb, json.dumps(meta))
                )
            self.conn.commit()
    
    def search(self, query, top_k=5):
        """Search using cosine similarity."""
        query_embedding = self._embed([query])[0]
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT text, 1 - (embedding <=> %s) AS similarity
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cur.fetchall()
        
        documents = [row[0] for row in results]
        scores = [row[1] for row in results]
        
        return documents, scores
```

### 6. Hybrid Search Implementation

```python
from rank_bm25 import BM25Okapi

class HybridVectorDB:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.documents = []
        self.bm25 = None
    
    def add_documents(self, documents):
        """Add documents to both vector and BM25 index."""
        self.documents.extend(documents)
        
        # Build BM25 index
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Add to vector DB
        self.vector_db.upsert_documents(documents)
    
    def search(self, query, top_k=10, alpha=0.5):
        """Hybrid search: alpha * vector + (1-alpha) * BM25."""
        # Vector search
        vector_docs, vector_scores = self.vector_db.search(query, top_k=top_k*2)
        
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        vector_scores_norm = np.array(vector_scores) / (max(vector_scores) + 1e-10)
        bm25_scores_norm = bm25_scores / (max(bm25_scores) + 1e-10)
        
        # Combine
        combined_scores = alpha * vector_scores_norm + (1 - alpha) * bm25_scores_norm
        
        # Get top K
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
```
