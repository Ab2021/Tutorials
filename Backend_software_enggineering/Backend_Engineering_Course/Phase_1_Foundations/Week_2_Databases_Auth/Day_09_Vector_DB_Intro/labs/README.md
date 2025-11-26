# Lab: Day 9 - Semantic Search with Vector DB

## Goal
Build a simple Semantic Search engine. You will use a local embedding model to convert text to vectors, store them in Qdrant, and query them.

## Prerequisites
- Docker (Qdrant).
- Python + `sentence-transformers` + `qdrant-client`.

## Directory Structure
```
day09/
├── docker-compose.yml
├── semantic_search.py
└── requirements.txt
```

## Step 1: Docker Compose

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

## Step 2: Requirements

```text
sentence-transformers
qdrant-client
torch
```

## Step 3: The Search Engine (`semantic_search.py`)

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import time

# 1. Load Model (Downloads ~100MB)
print("📥 Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "knowledge_base"

# 3. Create Collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Data
documents = [
    {"text": "The quick brown fox jumps over the lazy dog.", "category": "animals"},
    {"text": "Artificial Intelligence is transforming the world.", "category": "tech"},
    {"text": "Apples and Oranges are rich in Vitamin C.", "category": "food"},
    {"text": "Machine Learning models require data.", "category": "tech"},
    {"text": "Dogs are man's best friend.", "category": "animals"},
]

# 5. Embed & Index
print("🧠 Embedding and Indexing...")
points = []
for idx, doc in enumerate(documents):
    vector = model.encode(doc["text"]).tolist()
    points.append(PointStruct(id=idx, vector=vector, payload=doc))

client.upsert(collection_name=collection_name, points=points)
print(f"✅ Indexed {len(documents)} documents.")

# 6. Search Function
def search(query):
    print(f"\n🔎 Query: '{query}'")
    query_vector = model.encode(query).tolist()
    
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )
    
    for hit in hits:
        print(f" - Found: '{hit.payload['text']}' (Score: {hit.score:.4f})")

# 7. Test
search("healthy fruit")       # Should find Apples
search("neural networks")     # Should find Machine Learning
search("canine companion")    # Should find Dogs (even without the word 'dog')
```

## Step 4: Run It
1.  `docker-compose up -d`
2.  `pip install -r requirements.txt`
3.  `python semantic_search.py`

## Challenge
Modify the search to filter by category.
*Hint*: Use `query_filter` in `client.search`.
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue
# ...
query_filter=Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
)
```
