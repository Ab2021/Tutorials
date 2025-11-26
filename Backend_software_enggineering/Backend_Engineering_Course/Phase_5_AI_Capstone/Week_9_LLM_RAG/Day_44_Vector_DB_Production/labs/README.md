# Lab: Day 44 - Advanced Vector Search

## Goal
Implement Metadata Filtering with Qdrant.

## Prerequisites
- Docker (Qdrant).
- `pip install qdrant-client sentence-transformers`

## Step 1: The Code (`advanced_search.py`)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION = "products"

# 1. Create Collection
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 2. Data with Metadata
products = [
    {"id": 1, "text": "iPhone 15 Pro", "category": "electronics", "price": 999},
    {"id": 2, "text": "Samsung Galaxy S24", "category": "electronics", "price": 899},
    {"id": 3, "text": "Nike Air Max", "category": "clothing", "price": 120},
    {"id": 4, "text": "MacBook Pro", "category": "electronics", "price": 2000},
]

# 3. Index
points = []
for p in products:
    vector = model.encode(p["text"]).tolist()
    points.append(PointStruct(id=p["id"], vector=vector, payload=p))

client.upsert(collection_name=COLLECTION, points=points)

# 4. Filtered Search
# Find "phone" but ONLY in "electronics" category
query_vector = model.encode("phone").tolist()

search_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="electronics")
        )
    ]
)

hits = client.search(
    collection_name=COLLECTION,
    query_vector=query_vector,
    query_filter=search_filter,
    limit=3
)

print("--- Results (Electronics Only) ---")
for hit in hits:
    print(f"{hit.payload['text']} (Score: {hit.score:.2f})")
```

## Step 2: Run It
`python advanced_search.py`

*   **Observe**: It should find iPhone and Samsung. It should NOT find Nike (even if it was semantically similar, though "phone" and "shoe" are far apart).

## Challenge
Add a **Price Filter**.
Find "expensive device" where `price < 1000`.
*   *Hint*: Use `Range` condition in Qdrant filter.
