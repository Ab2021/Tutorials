# Lab 1: Hybrid Search System (Dense + Sparse)

## Objective
Combine **Semantic Search** (Vectors) with **Keyword Search** (BM25).
Vectors are bad at exact matches (e.g., "Error 503"). BM25 is great at it.
We will use **Reciprocal Rank Fusion (RRF)** to combine them.

## 1. Setup

```bash
poetry add rank_bm25 numpy scikit-learn
```

## 2. The Retriever (`hybrid.py`)

```python
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "The server returned Error 503.",
    "The capital of France is Paris.",
    "Python is a programming language.",
    "Error 503 means Service Unavailable."
]

# 1. Sparse Retriever (BM25)
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def sparse_search(query):
    scores = bm25.get_scores(query.split(" "))
    # Return list of (doc_idx, score)
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

# 2. Dense Retriever (Mock Vectors)
# In real life, use OpenAIEmbeddings
vectors = np.random.rand(len(documents), 128) 

def dense_search(query):
    query_vec = np.random.rand(1, 128) # Mock
    scores = cosine_similarity(query_vec, vectors)[0]
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

# 3. Reciprocal Rank Fusion (RRF)
def rrf(sparse_results, dense_results, k=60):
    fusion_scores = {}
    
    for rank, (doc_idx, _) in enumerate(sparse_results):
        if doc_idx not in fusion_scores: fusion_scores[doc_idx] = 0
        fusion_scores[doc_idx] += 1 / (k + rank + 1)
        
    for rank, (doc_idx, _) in enumerate(dense_results):
        if doc_idx not in fusion_scores: fusion_scores[doc_idx] = 0
        fusion_scores[doc_idx] += 1 / (k + rank + 1)
    
    return sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

# Run
query = "Error 503"
s_res = sparse_search(query)
d_res = dense_search(query)
final_res = rrf(s_res, d_res)

print("Top Document Index:", final_res[0][0])
print("Content:", documents[final_res[0][0]])
```

## 3. Analysis
*   BM25 should rank doc 0 and 3 high.
*   Vector search might be random (since we mocked it), but in reality, it would find semantically related docs.
*   RRF balances them.

## 4. Submission
Submit the code.
