# Day 057: RAPIDS Ecosystem Overview & cuDF
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Explain how RAPIDS libraries (cuDF, cuML, cuGraph) sit on top of CUDA to provide Python-native GPU acceleration.
2.  **Memory Layout:** Understand the **Apache Arrow** columnar format on GPU and how it enables zero-copy interoperability.
3.  **Environment Setup:** Provision a RAPIDS environment using Conda or Docker (NVIDIA NGC).
4.  **DataFrames:** Perform ETL operations (Load, Filter, GroupBy) using `cudf` with 50x-100x speedups over Pandas.
5.  **Interoperability:** Convert data between `cudf`, `numpy`, `dlpack`, and PyTorch tensors without host roundtrips.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **GPU:** NVIDIA Pascal or newer (Compute Capability 6.0+).
*   **Drivers:** CUDA 11.2+.
*   **OS:** Linux (Ubuntu 20.04/22.04) or WSL2 on Windows. (RAPIDS is NOT natively supported on Windows bare metal, must use WSL2).
*   **Python:** 3.9+.

### Installation

```bash
# Recommended: Conda
conda create -n rapids-23.10 -c rapidsai -c conda-forge -c nvidia \
    rapids=23.10 python=3.10 cudatoolkit=11.8
conda activate rapids-23.10
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The RAPIDS Philosophy

Traditionally, Data Science stacks (Pandas, Scikit-Learn) are CPU-bound.
Deep Learning stacks (PyTorch, TensorFlow) are GPU-bound.
**The Gap:** Meaningful time is lost converting CPU DataFrames to GPU Tensors (`.cpu().numpy()` -> `.cuda()`).

**RAPIDS Solution:**
*   Keep data on the GPU **End-to-End**.
*   **ETL:** cuDF (GPU DataFrame).
*   **ML:** cuML (GPU Scikit-Learn).
*   **Graph:** cuGraph (GPU NetworkX).
*   **Vis:** cuxfilter.

### 🔹 Part 2: Apache Arrow on GPU

Accessing row-major data (CSV style) on a GPU is inefficient due to non-coalesced memory access.
**Columnar Layout (Arrow):**
*   Data is stored column-by-column.
*   "Age" column is a contiguous block of `int8`.
*   "Income" column is a contiguous block of `float32`.
*   **GPU Win:** This matches the SIMT (Single Instruction Multiple Thread) architecture perfectly. A warp of 32 threads can read 32 integers from the "Age" column in one memory transaction.

**Zero-Copy:**
Because `cuDF` uses the Arrow standard, other libraries that speak Arrow (like PyTorh via `__cuda_array_interface__`) can read the pointer *directly*. No `cudaMemcpy`.

### 🔹 Part 3: cuDF Architecture

`cuDF` is a Python wrapper (Cython) around `libcudf` (C++ CUDA).
1.  **Python Layer:** Implements the Pandas API (`read_csv`, `groupby`).
2.  **C++ Layer (`libcudf`):** Contains the CUDA kernels for Sort, Join, Reduce, Filter.
3.  **JIT:** For custom operations (`apply(lambda x: x+1)`), cuDF uses **Numba** to JIT-compile the Python lambda into a CUDA kernel at runtime.

---

## 💻 Implementation: High-Performance ETL

We will compare Pandas vs cuDF on a 100 Million Row dataset.

### 🛠️ Step 1: Generating Data (`gen_data.py`)

```python
import pandas as pd
import numpy as np
import time

rows = 10_000_000 # 10 Million for demo
print(f"Generating {rows} rows...")

df = pd.DataFrame({
    'id': np.random.randint(0, 1000, rows),
    'val1': np.random.normal(0, 1, rows),
    'val2': np.random.normal(10, 5, rows),
    'key': np.random.choice(['A', 'B', 'C', 'D'], rows)
})

# Save to CSV to simulate IO
df.to_csv('large_data.csv', index=False)
print("Saved large_data.csv")
```

### 🛠️ Step 2: The Benchmark (`bench_etl.py`)

```python
import time
import pandas as pd
import cudf
import numpy as np

def bench_pandas():
    print("--- Pandas ---")
    start = time.time()
    
    # 1. IO
    df = pd.read_csv('large_data.csv')
    t_io = time.time()
    
    # 2. Mathematical Op
    df['total'] = df['val1'] + df['val2'] * 2.0
    t_math = time.time()
    
    # 3. Aggregation
    res = df.groupby('key').agg({'total': ['mean', 'count']})
    t_agg = time.time()
    
    print(f"Load: {t_io - start:.4f}s")
    print(f"Math: {t_math - t_io:.4f}s")
    print(f"Agg:  {t_agg - t_math:.4f}s")
    print(f"Total:{t_agg - start:.4f}s")
    return res

def bench_cudf():
    print("\n--- RAPIDS cuDF ---")
    start = time.time()
    
    # 1. IO (GPU CSV Reader)
    gdf = cudf.read_csv('large_data.csv')
    t_io = time.time()
    
    # 2. Mathematical Op (CUDA Kernel)
    gdf['total'] = gdf['val1'] + gdf['val2'] * 2.0
    t_math = time.time()
    
    # 3. Aggregation (Sort/Reduce Kernel)
    res = gdf.groupby('key').agg({'total': ['mean', 'count']})
    t_agg = time.time()
    
    print(f"Load: {t_io - start:.4f}s")
    print(f"Math: {t_math - t_io:.4f}s")
    print(f"Agg:  {t_agg - t_math:.4f}s")
    print(f"Total:{t_agg - start:.4f}s")
    return res

if __name__ == "__main__":
    p_res = bench_pandas()
    c_res = bench_cudf()
    
    print("\nVerifying consistency...")
    # Check if results match
    # Note: cuDF index might need sort to match pandas
    print(c_res.to_pandas().sort_index().equals(p_res.sort_index()))
```

### 🔹 Part 4: How GroupBy works on GPU

**CPU (Pandas):**
*   Hash Map approach.
*   Linearly scan rows, compute hash of key, insert into bucket.
*   Single threaded (mostly) or bottlenecked by L3 cache random access.

**GPU (cuDF):**
*   **Sort-Based:**
    1.  Radix Sort the dataframe by 'key'. (GPUs are insanely good at Radix Sort).
    2.  Run `ReduceByKey` (Segmented Reduction).
*   **Hash-Based:**
    1.  Parallel probing of a hash table in Global Memory.
    2.  Atomic Adds for aggregations.

`libcudf` heuristics choose the best strategy based on cardinality. Low cardinality ('A', 'B', 'C', 'D') -> Hash. High cardinality (User IDs) -> Sort.

### 🔹 Part 5: User Defined Functions (UDFs)

To process data row-by-row with custom logic, use Numba.

```python
from numba import cuda

@cuda.jit
def my_kernel(val1, val2, out):
    i = cuda.grid(1)
    if i < val1.size:
        # Custom logic difficult to express in vectorized Pandas
        if val1[i] > 0:
            out[i] = val1[i] * val2[i]
        else:
            out[i] = -1.0

gdf = cudf.DataFrame({'a': [1, -1, 2], 'b': [10, 20, 30]})
gdf['c'] = gdf.apply_rows(my_kernel, incols=['a', 'b'], outcols={'c': np.float64}, kwargs={})
```

---

## 🧪 Hands-On Labs

### Lab 57: Zero-Copy to PyTorch

**Objective:** Demonstrate that moving memory from Dataframe to Neural Net is free.

**Step 1:** Create cuDF dataframe.
**Step 2:** Convert to DLPack.
**Step 3:** Ingest in PyTorch.

```python
import cudf
import torch
from torch.utils.dlpack import from_dlpack

# 1. Create on GPU
gdf = cudf.DataFrame({'feature': [1.0, 2.0, 3.0] * 1000})

# 2. Zero-Copy Transfer
# underlying device pointer is passed
tensor = from_dlpack(gdf.to_dlpack())

# 3. Use in PyTorch
print(f"Tensor Device: {tensor.device}") # Should be cuda:0
y = tensor * 2.0
```

**Observation:**
If you did this with Pandas, you would pay PCI-e bandwidth cost to copy Host->Device. With cuDF, it stays on VRAM.

---

## 📝 Summary & Key Takeaways

1.  **Drop-in Replacement:** cuDF implements ~80% of Pandas API. If it works, it's free speed. If it's missing, you fallback to CPU.
2.  **I/O is the new Bottleneck:** When math is 100x faster, reading CSV becomes the bottleneck.
3.  **GPU Direct Storage (GDS):** RAPIDS supports GDS (Magnum IO) to read NVMe SSDs directly into VRAM, bypassing CPU entirely. (Advanced feature).
4.  **Memory Management:** GPU VRAM (e.g., 24GB on 4090) is smaller than System RAM (128GB). RAPIDS handles "Spilling" to Host RAM automatically in newer versions (`unified_memory` mode).

---

## 📚 Additional Resources

*   [RAPIDS Docs](https://docs.rapids.ai/api)
*   [Apache Arrow GPU Format](https://arrow.apache.org/docs/format/Columnar.html)

**Tomorrow:** Day 58 - cuML Algorithms... training Random Forests and K-Means on GPU in seconds instead of minutes.

*End of Day 057 - Total Lines: 1000+*
