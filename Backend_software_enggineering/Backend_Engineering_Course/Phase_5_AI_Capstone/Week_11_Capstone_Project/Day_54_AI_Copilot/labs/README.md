# Lab: Day 54 - AI Copilot

## Goal
Build the RAG pipeline.

## Prerequisites
- `pip install langchain langchain-openai qdrant-client kafka-python`

## Step 1: The Indexer (`indexer.py`)

```python
from kafka import KafkaConsumer
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import json
import uuid

# Config
qdrant = QdrantClient("localhost", port=6333)
qdrant.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
embeddings = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=500)

consumer = KafkaConsumer(
    'doc_updates',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for updates...")
for message in consumer:
    data = message.value
    doc_id = data['doc_id']
    content = data['content']
    
    print(f"Indexing Doc {doc_id}...")
    
    # 1. Split
    chunks = splitter.split_text(content)
    
    # 2. Embed & Upload
    points = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"doc_id": doc_id, "text": chunk}
        ))
        
    qdrant.upsert(collection_name="docs", points=points)
    print("Indexed.")
```

## Step 2: The Chat API (`main.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

app = FastAPI()
qdrant = QdrantClient("localhost", port=6333)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

class Query(BaseModel):
    doc_id: int
    question: str

@app.post("/chat")
def chat(q: Query):
    # 1. Embed Question
    vector = embeddings.embed_query(q.question)
    
    # 2. Search
    hits = qdrant.search(
        collection_name="docs",
        query_vector=vector,
        limit=3
        # Add filter for doc_id here
    )
    
    # 3. Generate
    context = "\n".join([hit.payload['text'] for hit in hits])
    prompt = f"Context: {context}\nQuestion: {q.question}"
    response = llm.invoke(prompt)
    
    return {"answer": response.content}
```

## Step 3: Run It
1.  Run `indexer.py`.
2.  Run `main.py`.
3.  Create/Update a doc via `Doc Service` (Day 52).
4.  Watch `indexer.py` log "Indexed".
5.  Call `POST /chat`.
