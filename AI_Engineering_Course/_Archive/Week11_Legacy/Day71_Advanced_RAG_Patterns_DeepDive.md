# Day 71: Advanced RAG Patterns
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. RAG Fusion Implementation (RRF)

Combining results from multiple query variations.

```python
import numpy as np

def generate_query_variations(llm, original_query):
    prompt = f"Generate 3 different search queries for: {original_query}"
    # response = llm.generate(prompt)
    return [original_query, "variation 1", "variation 2"]

def reciprocal_rank_fusion(results_list, k=60):
    """
    results_list: List of List of (doc_id, score)
    k: RRF constant
    """
    scores = {}
    
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in scores:
                scores[doc_id] = 0
            # RRF Formula
            scores[doc_id] += 1 / (k + rank + 1)
            
    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# Usage
# q_vars = generate_query_variations(llm, "climate change impact")
# all_results = [vector_db.search(q) for q in q_vars]
# fused_results = reciprocal_rank_fusion(all_results)
```

### 2. HyDE (Hypothetical Document Embeddings)

Generating a fake answer to search for real answers.

```python
def hyde_search(llm, embedding_model, vector_db, query):
    # 1. Generate Hypothetical Answer
    prompt = f"Write a short passage that answers the question: {query}"
    hypothetical_answer = llm.generate(prompt)
    print(f"Hypothetical: {hypothetical_answer}")
    
    # 2. Embed Hypothetical Answer
    query_vec = embedding_model.encode(hypothetical_answer)
    
    # 3. Search
    results = vector_db.search(query_vec)
    return results
```

### 3. Parent Document Retriever Logic

Indexing small chunks, retrieving large chunks.

```python
class ParentDocumentRetriever:
    def __init__(self):
        self.doc_store = {} # doc_id -> full_text
        self.vector_store = [] # (chunk_vec, doc_id)
        
    def add_document(self, doc_id, text):
        # Store full text
        self.doc_store[doc_id] = text
        
        # Split into small chunks
        chunks = self._split_text(text, chunk_size=100)
        
        # Embed and store chunks with pointer to parent
        for chunk in chunks:
            vec = self._embed(chunk)
            self.vector_store.append((vec, doc_id))
            
    def retrieve(self, query):
        query_vec = self._embed(query)
        
        # Find closest small chunks
        top_chunks = self._vector_search(query_vec)
        
        # Deduplicate parent IDs
        parent_ids = set(doc_id for _, doc_id in top_chunks)
        
        # Return full parent documents
        return [self.doc_store[pid] for pid in parent_ids]
```

### 4. GraphRAG Construction (Simple)

Building a knowledge graph from text.

```python
import networkx as nx

class SimpleGraphRAG:
    def __init__(self):
        self.graph = nx.Graph()
        
    def ingest(self, text, llm):
        # 1. Extract Entities & Relations using LLM
        prompt = f"Extract entities and relations from: {text}. Format: (Entity1, Relation, Entity2)"
        # response = llm.generate(prompt)
        # Mock response:
        triplets = [("Elon Musk", "CEO", "Tesla"), ("Tesla", "Makes", "EVs")]
        
        # 2. Build Graph
        for e1, rel, e2 in triplets:
            self.graph.add_edge(e1, e2, relation=rel)
            
    def query(self, entity):
        # 1. Find neighbors
        if entity not in self.graph:
            return []
        neighbors = self.graph[entity]
        
        # 2. Format context
        context = []
        for neighbor, attrs in neighbors.items():
            context.append(f"{entity} {attrs['relation']} {neighbor}")
            
        return context

# Usage
# kg = SimpleGraphRAG()
# kg.ingest("Elon Musk is CEO of Tesla.", llm)
# print(kg.query("Elon Musk"))
```

### 5. Self-RAG Critique Loop

Simulating the decision process.

```python
def self_rag_generate(llm, retriever, query):
    # 1. Retrieve
    docs = retriever.retrieve(query)
    
    # 2. Critique Relevance
    relevant_docs = []
    for doc in docs:
        score = llm.predict(f"Is this doc relevant to '{query}'? {doc}")
        if "yes" in score.lower():
            relevant_docs.append(doc)
            
    if not relevant_docs:
        return "No relevant info found."
        
    # 3. Generate
    context = "\n".join(relevant_docs)
    answer = llm.generate(f"Context: {context}\nQuestion: {query}")
    
    # 4. Critique Support
    support_score = llm.predict(f"Is the answer '{answer}' supported by context? {context}")
    
    if "yes" in support_score.lower():
        return answer
    else:
        return "Generated answer was not supported by text."
```
