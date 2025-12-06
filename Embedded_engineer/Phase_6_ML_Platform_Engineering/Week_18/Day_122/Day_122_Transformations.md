# Day 122: Map, Filter, Reduce: Distributed Transformations
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** ETL isn't just SQL queries. In AI, ETL involves running heavy neural networks (embedding generation). Ray Data handles both lightweight maps and heavyweight actor-based inference.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Task-based Map (`TaskPoolStrategy`) and Actor-based Map (`ActorPoolStrategy`).
2.  **Implement** `map_batches` to process data using a Hugging Face model.
3.  **Use** `flat_map` to explode 1 row into N rows.
4.  **Filter** rows based on a predicate.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install transformers torch`.

---

## 📖 Theoretical Foundation

### 1. The Strategy Decision
*   **Tasks (Default):** Ray starts a new Python worker, runs function, kills worker.
    *   *Pros:* No memory leaks. Easy scaling.
    *   *Cons:* Init overhead (importing `torch` takes 2s).
*   **Actors (Pool):** Ray starts N actors. Keeps them alive. Feeds batches to them.
    *   *Pros:* Zero init overhead after startup.
    *   *Cons:* State management.

### 2. Map Batches
Always use `map_batches` instead of `map`. Vectorization (Pandas/Arrow) is 100x faster than Python loops.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Actor-based Mapping

We process text using a generic HF Tokenizer.

#### 📁 `src/03_transform.py`
```python
import ray
import pandas as pd
from transformers import AutoTokenizer

# 1. Create Data
df = pd.DataFrame({"text": ["Hello World"] * 1000})
ds = ray.data.from_pandas(df)

# 2. Define Class for Actor
class Tokenizer:
    def __init__(self):
        # Expensive init (runs once)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, batch):
        # Batch is Dict[str, np.array] or pd.DataFrame
        texts = batch["text"].tolist()
        tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="np")
        return {"input_ids": tokens["input_ids"]}

# 3. Apply Transform
# Use ActorPool to reuse the Tokenizer object
ds_tokenized = ds.map_batches(
    Tokenizer, 
    compute=ray.data.ActorPoolStrategy(size=2), # 2 Actors
    batch_size=32
)

# 4. Trigger
print(ds_tokenized.take(1))
```

### 👨‍💻 Core Implementation: FlatMap

Splitting sentences.

```python
def split_sentences(row):
    # Input row: {"text": "A. B. C."}
    # Output list: [{"sent": "A"}, {"sent": "B"}, ...]
    sentences = row["text"].split(".")
    return [{"sentence": s.strip()} for s in sentences if s.strip()]

ds_split = ds.flat_map(split_sentences)
```

---

## 🔬 Lab Exercise: "Init overhead"

### Task
Measure the difference.
1.  Define a function `slow_init(batch)`. It does `time.sleep(1)` (simulating import) then returns batch.
2.  Run with `map_batches(slow_init)`.
    *   Wait time: 1s per task? (Actually Ray reuses worker processes loosely, but imports might persist).
3.  Run with `ActorPoolStrategy`.
    *   Observer: First batch slow. Subsequent batches fast.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Batch Size:** Tunable parameter. Bigger is better for GPU throughput, but consumes RAM.
2.  **Ordering:** `map_batches` does NOT preserve order by default (unless `preserve_order=True`). DL training usually doesn't care about global order (we shuffle anyway).
3.  **Generator:** `flat_map` allows data expansion (e.g., Data Augmentation: 1 Image -> 5 Crops).

### API Summary
```python
ds.map_batches()
ds.flat_map()
ds.filter()
```

---

**Day 122 Complete** ✅

*Next: Day 123 - Shuffle & GroupBy - The Hardest Problem in Distributed Systems.*
