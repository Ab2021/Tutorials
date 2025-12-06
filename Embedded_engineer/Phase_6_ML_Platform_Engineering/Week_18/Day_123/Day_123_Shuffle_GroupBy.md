# Day 123: The Shuffle: Sorting at Scale
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** Shuffling is the most expensive operation in distributed computing. It saturates the network switch. Learn when to use **Global Shuffle**, **Local Shuffle**, or avoid it entirely.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Map-Reduce shuffle mechanism (Partitioning phases).
2.  **Execute** a distributed `groupby` and aggregation.
3.  **Perform** a `random_shuffle` for ML training data.
4.  **Tune** the number of reduce tasks (partitions).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. Global Shuffle (Sort/GroupBy)
To group by "User ID":
1.  **Map:** Each block calculates separate counts. Hashes "User ID" to assign to Reducer N.
2.  **Shuffle:** Map tasks send data to Reducer tasks via Object Store (Network).
3.  **Reduce:** Reducer N aggregates counts for its User IDs.

### 2. Random Shuffle (ML)
Training needs random batches.
*   **Ray Data:** Uses a specialized shuffle (Push-based).
*   **Windowing:** If dataset is too huge, use `random_shuffle(window_size=...)` to only shuffle chunks fitting in memory.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: GroupBy

#### 📁 `src/04_shuffle.py`
```python
import ray
import pandas as pd

# 1. Create Data (with duplicates)
df = pd.DataFrame({
    "category": ["A", "B", "A", "C", "B"] * 1000,
    "value": range(5000)
})
ds = ray.data.from_pandas(df)

# 2. GroupBy Count
grouped = ds.groupby("category").count()
print(grouped.take_all())
# [{'category': 'A', 'count': 2000}, ...]

# 3. Map Groups
# Apply logic per group (e.g., Normalize value within category)
def normalize(group):
    # group comes as a Dict of arrays
    mean = group["value"].mean()
    group["normalized"] = group["value"] - mean
    return group

result = ds.groupby("category").map_groups(normalize)
print(result.take(5))
```

### 👨‍💻 Core Implementation: Random Shuffle

```python
# Fully global shuffle
shuffled = ds.random_shuffle()

# Save to disk to lock it in
# shuffled.write_parquet("/tmp/shuffled_data")
```

---

## 🔬 Lab Exercise: "OOM on Shuffle"

### Task
Scale limits.
1.  Try to sort a dataset larger than your RAM.
2.  Ray *should* spill to disk.
3.  Monitor `ray memory` or dashboard "Disk Usage" during the shuffle.
4.  If it crashes, increase object store memory limits or enable spilling explicitly.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Training:** For training, `random_shuffle` is crucial after every epoch (if streaming from cache) to ensure convergence.
2.  **Cost:** Shuffle cost grows quadratically with number of nodes (network connections). Minimize shuffles.
3.  **Partitions:** Ensure you have enough partitions (blocks) to leverage all CPUs during reduce phase.

### API Summary
```python
ds.random_shuffle()
ds.groupby()
```

---

**Day 123 Complete** ✅

*Next: Day 124 - Custom Datasources - Reading proprietary formats.*
