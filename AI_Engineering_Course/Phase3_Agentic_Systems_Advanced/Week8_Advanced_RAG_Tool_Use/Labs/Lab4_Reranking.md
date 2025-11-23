# Lab 4: Cross-Encoder Reranking

## Objective
Bi-Encoders (Vector DB) are fast but less accurate.
**Cross-Encoders** are slow but very accurate.
Use Bi-Encoder for retrieval (Top-100), Cross-Encoder for reranking (Top-5).

## 1. The Reranker (`rerank.py`)

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "Who wrote Hamlet?"
docs = [
    "Hamlet was written by Shakespeare.", # Relevant
    "Harry Potter was written by Rowling.", # Irrelevant
    "Shakespeare lived in Stratford." # Semi-relevant
]

# Score pairs
pairs = [[query, doc] for doc in docs]
scores = model.predict(pairs)

# Sort
ranked = sorted(zip(scores, docs), reverse=True)

for score, doc in ranked:
    print(f"{score:.4f}: {doc}")
```

## 2. Analysis
The Cross-Encoder sees the query and document *together*, allowing it to capture deep semantic interactions.

## 3. Submission
Submit the scores for the 3 documents.
