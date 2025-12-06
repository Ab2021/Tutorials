# Day 120: The Last Mile: Ray Data Architecture
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** Spark is great for ETL (SQL), but terrible for ML (Tensors). **Ray Data** bridges the gap, allowing you to stream Petrabyte-scale datasets directly into PyTorch/TensorFlow with native Python code.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Ray Data (Streaming) with Spark (Bulk/Stage-based).
2.  **Create** a distributed Dataset from memory, files, and S3.
3.  **Inspect** the Block-based architecture (Arrow Tables).
4.  **Visualize** the execution plan of a lazy dataset.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install "ray[data]" pyarrow pandas`.

---

## 📖 Theoretical Foundation

### 1. The ML I/O Problem
*   **ETL Phase:** Often done in Spark/Snowflake. Output: Parquet files.
*   **Training Phase:** `DataLoader` must read Parquet, decode, shuffle, batch, and send to GPU.
*   **Ray Data:** Replaces the `DataLoader`. It reads Parquet distributedly and streams batches to the GPU.

### 2. Physical Layout (Blocks)
*   A `Dataset` is a list of References to **Blocks**.
*   Blocking factor: 1 Block $\approx$ 512MB (Configurable).
*   Format: **Apache Arrow** (Columnar, Zero-Copy).

### 3. Execution Engine (Streaming)
*   **Bulk (Spark):** Map -> Barrier -> Reduce -> Barrier -> Write. High latency.
*   **Streaming (Ray):** Map task 1 finishes -> Reduce task 1 starts immediately. Overlaps Compute and I/O.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Creating Datasets

#### 📁 `src/01_basics.py`
```python
import ray
import pandas as pd
import numpy as np

# 1. From Laptop Memory (Small)
# Ray splits this into blocks distributed across the cluster (if connected)
df = pd.DataFrame({"a": np.random.randn(1000), "b": np.random.randn(1000)})
ds = ray.data.from_pandas(df)

print(ds)
# Dataset(
#    num_blocks=...,
#    num_rows=1000,
#    schema={a: double, b: double}
# )

# 2. From Storage (Lazy)
# Does NOT read the file yet. Just lists metadata.
# ds = ray.data.read_parquet("s3://my-bucket/data.parquet")

# 3. Consumption (Triggers execution)
# Take first 5 rows
print(ds.take(5))

# 4. Schema Inspection
print(ds.schema())
```

### 👨‍💻 Core Implementation: Block Inspection

You can see where the data lives.

```python
# usage_stats shows memory usage per node
print(ds.stats())

# Access low-level blocks
# (Each block is an ObjectRef pointing to an Arrow Table)
for block_ref in ds.get_internal_block_refs():
    meta = ray.data.context.DataContext.get_current().execution_options
    print(block_ref)
```

---

## 🔬 Lab Exercise: "Arrow vs Pandas"

### Task
Measure Overhead.
1.  Create a Dataset with 1M rows.
2.  `ds.map_batches(lambda batch: batch + 1, batch_format="pandas")`.
    *   Ray must convert Arrow -> Pandas -> Arrow.
3.  `ds.map_batches(..., batch_format="pyarrow")`.
    *   Zero copy (if operation supports Arrow).
4.  **Observation:** Pyarrow is 10x faster for simple arithmetic. Use Pandas only if you need complex logic not supported by Arrow/Numpy.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Lazy:** Like Spark, transformations are lazy. `read` + `map` + `filter` builds a plan. `take`, `show`, `write`, or `iter_batches` executes it.
2.  **Unstructured Data:** Ray Data shines with Images/Audio (Numpy blocks), whereas Spark struggles with non-tabular data.
3.  **Global Shuffle:** Ray 2.0+ introduced a highly optimized Shuffle for sorting 100TB+ datasets, rivaling Spark's sort performance.

### API Summary
```python
ray.data.range(N)
ray.data.read_csv()
ds.take_batch()
```

---

**Day 120 Complete** ✅

*Next: Day 121 - Connectors - Reading from the World.*
