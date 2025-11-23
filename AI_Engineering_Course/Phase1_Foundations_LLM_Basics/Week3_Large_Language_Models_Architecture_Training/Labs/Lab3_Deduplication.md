# Lab 3: Data Deduplication (MinHash LSH)

## Objective
Training on duplicate data hurts performance.
We will implement **MinHash LSH** (Locality Sensitive Hashing) to find near-duplicates.

## 1. Setup
```bash
pip install datasketch
```

## 2. The Pipeline (`dedup.py`)

```python
from datasketch import MinHash, MinHashLSH
import re

data = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog!", # Near duplicate
    "Machine learning is fascinating.",
    "Deep learning is fascinating."
]

def get_minhash(text):
    tokens = set(text.lower().split())
    m = MinHash(num_perm=128)
    for t in tokens:
        m.update(t.encode('utf8'))
    return m

# 1. Index
lsh = MinHashLSH(threshold=0.5, num_perm=128)
minhashes = {}

for i, text in enumerate(data):
    m = get_minhash(text)
    minhashes[i] = m
    lsh.insert(f"doc_{i}", m)

# 2. Query
for i, text in enumerate(data):
    result = lsh.query(minhashes[i])
    print(f"Doc {i} matches: {result}")
```

## 3. Analysis
*   Doc 0 and 1 should match.
*   Doc 2 and 3 might match depending on threshold.

## 4. Submission
Submit the output showing the detected duplicates.
