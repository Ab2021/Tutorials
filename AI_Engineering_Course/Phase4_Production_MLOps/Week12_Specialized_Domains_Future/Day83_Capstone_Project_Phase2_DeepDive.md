# Day 83: Capstone Project Phase 2 - Implementation (MVP)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Ingestion Script (The "Loader")

Parsing and Indexing.

```python
from unstructured.partition.pdf import partition_pdf
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import openai
import uuid

# Setup
client = QdrantClient("localhost", port=6333)
client.recreate_collection(
    collection_name="capstone_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

def ingest_pdf(filepath, doc_id):
    print(f"Parsing {filepath}...")
    elements = partition_pdf(filepath)
    
    chunks = []
    current_chunk = ""
    
    # Simple Chunking
    for el in elements:
        text = str(el)
        if len(current_chunk) + len(text) < 1000:
            current_chunk += " " + text
        else:
            chunks.append(current_chunk)
            current_chunk = text
    chunks.append(current_chunk)
    
    print(f"Embedding {len(chunks)} chunks...")
    points = []
    for i, chunk in enumerate(chunks):
        # Embed
        resp = openai.embeddings.create(input=chunk, model="text-embedding-3-small")
        vec = resp.data[0].embedding
        
        # Payload
        payload = {
            "text": chunk,
            "doc_id": doc_id,
            "chunk_index": i
        }
        
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
        
    # Upsert
    client.upsert(collection_name="capstone_docs", points=points)
    print("Done.")

# Usage
# ingest_pdf("annual_report.pdf", "doc_1")
```

### 2. Retrieval & Generation (The "Bot")

Simple RAG function.

```python
def chat_with_docs(query):
    # 1. Embed Query
    resp = openai.embeddings.create(input=query, model="text-embedding-3-small")
    q_vec = resp.data[0].embedding
    
    # 2. Search
    hits = client.search(
        collection_name="capstone_docs",
        query_vector=q_vec,
        limit=3
    )
    
    # 3. Construct Context
    context = "\n\n".join([hit.payload['text'] for hit in hits])
    
    # 4. Generate
    prompt = f"""
    Answer the question based on the context.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content

# Usage
# print(chat_with_docs("What was the revenue?"))
```

### 3. FastAPI Wrapper

Exposing as an API.

```python
from fastapi import FastAPI, UploadFile, File
import shutil

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger ingestion (Sync for MVP)
    ingest_pdf(path, file.filename)
    return {"status": "indexed"}

@app.post("/chat")
async def chat(query: str):
    response = chat_with_docs(query)
    return {"response": response}
```

### 4. Docker Compose Setup

Infrastructure as Code.

```yaml
version: '3'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
```
