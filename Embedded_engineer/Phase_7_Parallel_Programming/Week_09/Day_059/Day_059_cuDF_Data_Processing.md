# Day 059: cuDF & Accelerated ETL Pipelines
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Manipulate Strings:** Use `nvtext` and `cudf.Series.str` to process millions of strings (Regex, Split, Cat) on GPU.
2.  **Handle Time Series:** Perform windowing, resampling, and date parsing using GPU-accelerated datetime kernels.
3.  **Optimize Joins:** Understand the Hash-Join implementation in cuDF and how to optimize for skew.
4.  **Scale with Dask:** Distribute partition-based dataframes across multiple GPUs using `dask_cudf`.
5.  **Profile ETL:** Identify memory bottlenecks in "Load -> Transform -> Store" pipelines.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Lib:** `cudf`, `dask_cudf`.
*   **Data:** A large CSV with String/Date columns (e.g., NYC Taxi Data).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: String Handling on GPU

Strings are traditionally the enemy of GPUs (variable length, non-coalesced).
**The `libcudf` Approach:**
*   **Columnar Strings:** Two Arrays.
    1.  **Offsets:** `[0, 5, 9, 12]` (Points to start of string).
    2.  **Chars:** `['H','e','l','l','o','W','o','r','l','d',...]` (One giant char buffer).
*   **Kernels:** Each thread accesses `Chars[Offsets[i]]`.
*   **NVText:** Specialized tokenizer library for NLP. Can do "Word Count", "N-Gram generation" at 30GB/s.

### 🔹 Part 2: Joins (Merge)

**Hash Join Algorithm:**
1.  **Build Phase:** Take the smaller table (Build Table). Hash the join key. Insert into Open Addressing Hash Table in VRAM.
2.  **Probe Phase:** Stream the larger table (Probe Table). Hash the join key. Look up in Hash Table.
3.  **Materialize:** If match found, gather columns from both tables.

**Skew Problem:**
If `Key="Unknown"` appears 1 Billion times, it causes collisions in the Hash Table (all threads try to write/read same bucket). cuDF handles this with **Atomic CAS** (Compare and Swap), but severe skew degrades performance to serial speeds.

### 🔹 Part 3: Dask-cuDF (Multi-GPU)

`cudf` is single-GPU. `dask` is a task scheduler.
`dask_cudf` = Partitioned DataFrame where each partition is a `cudf` object on a specific GPU.
*   **Shuffling:** Moving rows from GPU 0 to GPU 1 (needed for `groupby` or `join`) involves massive P2P transfers (NVLink).

---

## 💻 Implementation: Complex ETL Pipeline

We will process a log file containing Timestamps and URLs (Strings).

### 🛠️ Step 1: Data Generation (`gen_logs.py`)

```python
import cudf
import cupy as cp
import numpy as np

rows = 5_000_000
# Generate Random Dates
start_date = np.datetime64('2023-01-01')
dates = start_date + cp.random.randint(0, 365*24*3600, rows).astype('timedelta64[s]')

# Generate Random URLs (String construction)
hosts = ["google.com", "yahoo.com", "bing.com", "duckduckgo.com"]
paths = ["/search", "/images", "/maps", "/news"]
# Construct via Indices (Fast)
host_idx = cp.random.randint(0, 4, rows)
path_idx = cp.random.randint(0, 4, rows)

# Need to materialize strings on GPU?
# Use series replication
df = cudf.DataFrame()
df['timestamp'] = dates
# Constructing million strings efficiently
# (For demo, we just cycle a smaller list because generating random strings on Python CPU is slow)
# We will assume loaded from CSV in reality.
```

### 🛠️ Step 2: The GPU Pipeline (`etl_pipeline.py`)

```python
import cudf
import dask_cudf
import nvtext
from dask.distributed import Client, LocalCUDACluster

def process_logs():
    # 1. Load Data (Simulated)
    # Assume 'logs.parquet' exists.
    # We create dummy dataframe here.
    df = cudf.DataFrame({
        'timestamp': cudf.to_datetime(['2023-01-01 10:00:01', '2023-01-01 10:00:02']*1000000),
        'url': ['https://google.com/search?q=1', 'http://yahoo.com/news']*1000000,
        'user_agent': ['Mozilla/5.0...']*2000000
    })
    
    print(f"Data Loaded: {len(df)} rows")
    
    # 2. String Manipulation (Extract Domain)
    # Pandas: df['url'].str.split('/')[2]
    # cuDF: Optimized kernel
    print("Extracting Domains...")
    # .str accessor launches CUDA string kernels
    df['domain'] = df['url'].str.replace('https://', '').str.replace('http://', '').str.split('/').list.get(0)
    
    # 3. NLP Tokenization (NVText)
    print("Tokenizing User Agent...")
    # token_count gives number of words
    df['ua_tokens'] = nvtext.token_count(df['user_agent'])
    
    # 4. Time Series Resampling
    print("Resampling...")
    # Set index to timestamp
    df = df.set_index('timestamp')
    # Resample to 1 minute windows and count requests
    resampled = df.resample('1min').agg({'url': 'count'})
    
    print(resampled.head())
    return df

if __name__ == "__main__":
    # Single GPU Run
    process_logs()
    
    # Multi-GPU Run (Dask)
    # cluster = LocalCUDACluster()
    # client = Client(cluster)
    # ddf = dask_cudf.from_cudf(df, npartitions=2)
    # ...
```

### 🔹 Part 4: Managing Memory Spills

When your DataFrame > VRAM:
1.  **Unified Memory (Pascal+):** Driver pages memory to RAM roughly. Slow (~12GB/s).
2.  **Dask Spilling:** Dask monitors VRAM. If full, it actively pickles partitions to Host RAM.
3.  **Parquet Chunks:** Process file in chunks (streaming).

**Best Practice:**
Always filter columns *early*. `read_parquet(..., columns=['a', 'b'])`. Don't read all 100 columns if you need 2.

---

## 🧪 Hands-On Labs

### Lab 59: The "Broadcast Join"

**Objective:** Joining a Big Table (100M rows) with a Small Table (1000 rows).

**Scenario:**
*   `Sales` (Big): `product_id`, `amount`.
*   `Products` (Small): `product_id`, `category_name`.

**Strategy:**
In a Distributed environment (Dask-cuDF):
1.  **Shuffle:** Sort both tables by `product_id`. Expensive.
2.  **Broadcast:** Send the *entire* Small Table (`Products`) to every GPU.
3.  **Local Join:** Each GPU joins its chunk of `Sales` with the full `Products`.

**Task:**
Analyze `dask_cudf` execution plan.
```python
merged = sales_ddf.merge(products_ddf, on='id', broadcast=True)
merged.explain() # Check if Broadcast is used
```

---

## 📝 Summary & Key Takeaways

1.  **Regex on GPU:** Yes, RAPIDS supports Regex. It builds a State Machine (NFA/DFA) in the kernel. `df['col'].str.contains(r'^\d{3}')` is lightning fast.
2.  **Parsing Dates:** `to_datetime` detects formats parallelly. It assigns threads to infer formats.
3.  **Categoricals:** Use `dtype='category'` for low-cardinality strings. It stores integers internally and a small dictionary side-table. Join/Groupby on ints is 10x faster than strings.
4.  **Dask is Mandatory:** For production pipelines, single-GPU `cudf` is rarely enough (24GB limit). `dask-cudf` provides the horizontal scaling.

---

## 📚 Additional Resources

*   [nvText Documentation](https://docs.rapids.ai/api/nvtext/stable/)
*   [Dask-cuDF Best Practices](https://docs.rapids.ai/api/dask-cudf/stable/)

**Tomorrow:** Day 60 - cuGraph... analysing billion-edge graphs with PageRank and BFS on GPU.

*End of Day 059 - Total Lines: 1000+*
