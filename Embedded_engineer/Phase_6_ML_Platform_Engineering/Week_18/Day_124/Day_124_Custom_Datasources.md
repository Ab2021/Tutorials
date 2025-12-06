# Day 124: Beyond Parquet: Custom Datasources
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** Your company stores data in a legacy binary format from 1995. Instead of converting 1PB to Parquet, write a **Custom Ray Datasource** to read it natively.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Subclass** `ray.data.Datasource`.
2.  **Implement** `get_read_tasks` to partition the read job.
3.  **Define** a generator function to yield Arrow tables from raw bytes.
4.  **Register** and use the new datasource.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Contract
Ray needs to know:
1.  **Parallelism:** How to split the job? (List of file paths).
2.  **Schema:** What columns will come out?
3.  **Data:** A function to actually read a split.

### 2. High Level API (Simple)
For file-based sources, use `ray.data.read_binary_files` and pass a custom decoder.
For complex sources (API polling), implement `Datasource`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Custom Reader

Let's assume we have files where every line is `ID|VALUE` (pipe separated).

#### 📁 `src/05_custom_source.py`
```python
import ray
from ray.data.datasource import Datasource, ReadTask
import pyarrow as pa
import os

# 1. Create Dummy Data
os.makedirs("data", exist_ok=True)
with open("data/file1.pipe", "w") as f:
    f.write("1|100\n2|200\n")
with open("data/file2.pipe", "w") as f:
    f.write("3|300\n4|400\n")

# 2. Define Logic to read a single file
def read_pipe_file(files):
    # files is a list of paths
    ids = []
    values = []
    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                ids.append(int(parts[0]))
                values.append(int(parts[1]))
    
    # Return Arrow Table
    return pa.Table.from_pydict({"id": ids, "value": values})

# 3. Use built-in generic file reader with custom function
# This is the modern (Ray 2.5+) way for file-based sources
ds = ray.data.read_binary_files(
    "data/*.pipe",
    override_num_blocks=2,
)

# read_binary_files gives us raw bytes. We map it.
def parse_bytes(row):
    # row is {"bytes": b"..."}
    text = row["bytes"].decode("utf-8")
    lines = text.strip().split("\n")
    ret = []
    for line in lines:
        parts = line.split("|")
        ret.append({"id": int(parts[0]), "value": int(parts[1])})
    return ret

ds_parsed = ds.flat_map(parse_bytes)
print(ds_parsed.take_all())
```

### 👨‍💻 Advanced: Full Datasource Implementation

If you need to connect to a socket or unconventional source.

```python
class RandomDatasource(Datasource):
    def get_read_tasks(self, parallelism):
        meta = ...
        
        def _read_fn():
             yield pa.Table.from_array(...)
             
        tasks = [ReadTask(_read_fn, meta) for _ in range(parallelism)]
        return tasks

# Usage
ds = ray.data.read_datasource(RandomDatasource(), parallelism=4)
```

---

## 🔬 Lab Exercise: "Metadata Only"

### Task
Implement a Reader that only reads headers.
1.  Goal: Scan 1KB header of 1GB Video Files.
2.  Use `read_binary_files(include_paths=True)`.
3.  In map function, open path, `f.seek(0)`, read 1024 bytes.
4.  Return Metadata Dict.
5.  **Result:** Fast indexing of massive binary blobs.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Simplicity:** Prefer `read_binary_files` + `flat_map` over subclassing `Datasource` unless strictly necessary (like API pagination).
2.  **Streaming:** Ensure your read function yields blocks/batches, don't read the whole 100GB file into RAM before yielding.

### API Summary
```python
ray.data.read_datasource()
```

---

**Day 124 Complete** ✅

*Next: Day 125 - Performance Tuning - Handling OOM and Backpressure.*
