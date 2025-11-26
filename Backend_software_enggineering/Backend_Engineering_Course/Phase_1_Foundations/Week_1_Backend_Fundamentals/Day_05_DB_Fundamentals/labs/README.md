# Lab: Day 5 - Polyglot Persistence Stack

## Goal
Spin up a multi-database environment using Docker Compose. You will connect to **Postgres** (SQL), **Redis** (Cache), and **Qdrant** (Vector DB) from a single Python script.

## Prerequisites
- Docker & Docker Compose.
- Python 3.11+.

## Directory Structure
```
day05/
├── docker-compose.yml
├── requirements.txt
└── main.py
```

## Step 1: Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'
services:
  # 1. Relational DB
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp
    ports:
      - "5432:5432"

  # 2. Key-Value Cache
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # 3. Vector DB
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

## Step 2: Python Dependencies (`requirements.txt`)

```text
psycopg2-binary
redis
qdrant-client
```

## Step 3: The Polyglot Script (`main.py`)

```python
import psycopg2
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import time

def run_postgres():
    print("\n🐘 --- Postgres (Relational) ---")
    conn = psycopg2.connect("dbname=myapp user=user password=password host=localhost")
    cur = conn.cursor()
    
    # Create Table
    cur.execute("CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, name TEXT);")
    
    # Insert
    cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("Alice",))
    user_id = cur.fetchone()[0]
    print(f"✅ Inserted User ID: {user_id}")
    
    conn.commit()
    cur.close()
    conn.close()

def run_redis():
    print("\n🔴 --- Redis (Key-Value) ---")
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Set Cache
    r.set('session:user_1', 'active', ex=60) # Expires in 60s
    val = r.get('session:user_1')
    print(f"✅ Retrieved from Cache: {val}")

def run_qdrant():
    print("\n🟣 --- Qdrant (Vector) ---")
    client = QdrantClient("localhost", port=6333)
    
    collection_name = "demo_collection"
    
    # Re-create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    
    # Upsert Vectors (Embeddings)
    # Imagine these are embeddings for ["Apple", "Banana", "Car"]
    vectors = [
        PointStruct(id=1, vector=[0.9, 0.1, 0.1, 0.2], payload={"name": "Apple"}),
        PointStruct(id=2, vector=[0.8, 0.2, 0.1, 0.1], payload={"name": "Banana"}),
        PointStruct(id=3, vector=[0.1, 0.1, 0.9, 0.8], payload={"name": "Car"}),
    ]
    client.upsert(collection_name=collection_name, points=vectors)
    print("✅ Indexed 3 vectors.")
    
    # Search: "Something like a fruit" (Vector close to Apple/Banana)
    query_vector = [0.85, 0.15, 0.1, 0.1] 
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )
    
    print("🔍 Search Results (Nearest Neighbors):")
    for hit in hits:
        print(f" - {hit.payload['name']} (Score: {hit.score:.4f})")

if __name__ == "__main__":
    # Wait for containers to boot
    print("Waiting for containers...")
    time.sleep(5) 
    
    try:
        run_postgres()
        run_redis()
        run_qdrant()
    except Exception as e:
        print(f"❌ Error: {e}")
```

## Step 4: Run It

1.  **Start Stack**:
    ```bash
    docker-compose up -d
    ```
2.  **Install Libs**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Script**:
    ```bash
    python main.py
    ```

## Expected Output
You should see successful interactions with all three databases, proving you can orchestrate a complex data layer.
