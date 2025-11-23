# Lab 3: Vector Memory Module

## Objective
Give your agent long-term memory.
We will create a `Memory` class wrapping ChromaDB.

## 1. The Module (`memory.py`)

```python
import chromadb

class Memory:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("agent_memory")

    def add(self, text, metadata={}):
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(hash(text))]
        )

    def retrieve(self, query, k=1):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]

# Usage
mem = Memory()
mem.add("User likes pizza.")
mem.add("User lives in NYC.")

print(mem.retrieve("What does the user eat?"))
```

## 2. Challenge
Add **Time-decay**. Retrieve memories based on relevance AND recency.

## 3. Submission
Submit the code for the time-decay retrieval.
