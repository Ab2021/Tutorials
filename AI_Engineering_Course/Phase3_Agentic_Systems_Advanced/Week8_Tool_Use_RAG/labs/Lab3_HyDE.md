# Lab 3: Query Expansion (HyDE)

## Objective
Queries are short. Documents are long.
**HyDE** (Hypothetical Document Embeddings) generates a fake document to bridge the gap.

## 1. The Pipeline (`hyde.py`)

```python
# 1. Query
query = "How to bake a cake?"

# 2. Generator (LLM)
def generate_hypothetical(query):
    # Mock LLM output
    return f"To bake a cake, you need flour, sugar, and eggs. Mix them..."

hypo_doc = generate_hypothetical(query)

# 3. Encoder
# embedding = model.encode(hypo_doc)

# 4. Search
# results = vector_db.search(embedding)

print(f"Query: {query}")
print(f"Hypothetical Doc: {hypo_doc}")
```

## 2. Analysis
The `hypo_doc` contains keywords (flour, sugar) that might not be in the query but ARE in the target documents.
This improves retrieval recall.

## 3. Submission
Submit a hypothetical document generated for the query "What is the capital of the moon?".
