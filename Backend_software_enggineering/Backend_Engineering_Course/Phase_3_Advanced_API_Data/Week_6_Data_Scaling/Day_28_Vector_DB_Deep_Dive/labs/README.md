# Lab: Day 28 - Semantic Search with Qdrant

## Goal
Build a search engine that understands meaning, not just keywords.

## Prerequisites
- Docker (Qdrant).
- `pip install qdrant-client sentence-transformers`

## Step 1: Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Step 2: The Code (`search.py`)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Init
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2') # Small, fast model

COLLECTION_NAME = "my_knowledge_base"

# 2. Create Collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 3. Data
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn animal leaps over a sleepy canine.", # Semantically identical to above
    "I love programming in Python.",
    "Machine learning is fascinating.",
    "Delicious pizza recipes with cheese.",
]

# 4. Embed & Index
print("Generating embeddings...")
vectors = model.encode(documents)

points = [
    PointStruct(id=idx, vector=v.tolist(), payload={"text": doc})
    for idx, (doc, v) in enumerate(zip(documents, vectors))
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(documents)} documents.")

# 5. Search
query_text = "fast animal dog"
print(f"\nQuery: '{query_text}'")

query_vector = model.encode(query_text).tolist()
hits = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

for hit in hits:
    print(f"Score: {hit.score:.4f} | Text: {hit.payload['text']}")
```

## Step 3: Run It
`python search.py`

*   **Observe**:
    *   Query "fast animal dog" should match "The quick brown fox..." AND "A fast auburn animal..." with high scores.
    *   It should NOT match "Delicious pizza".

## Challenge
Add a filter.
1.  Add `payload={"text": doc, "category": "tech"}` to some docs.
2.  Modify the search to only return results where `category == "tech"`.
