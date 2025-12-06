# Day 86: Zero-Copy Speed: Ray Object Store
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** In ML, passing 10GB DataFrames between processes kills performance. Ray's **Plasma Object Store** allows workers to access shared data with **Zero-Copy** overhead.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the concept of Shared Memory vs Inter-Process Communication (IPC).
2.  **Use** `ray.put()` to store large objects in Plasma effectively.
3.  **Demonstrate** Zero-Copy reads with NumPy arrays using Apache Arrow.
4.  **Debug** memory leaks using `ray memory` CLI.
5.  **Differentiate** between Passing by Value (Small args) vs Passing by Reference (ObjectRefs).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Machine with >4GB RAM.

### Software Environment
- `pip install numpy pandas pyarrow`.

---

## 📖 Theoretical Foundation

### 1. The Serialization Tax
Typical Python (multiprocessing):
1.  Process A has object.
2.  Pickle it (CPU heavy).
3.  Write to pipe.
4.  Process B reads pipe.
5.  Unpickle it (CPU heavy).
6.  **Result:** 2 copies of data. High Latency.

### 2. The Ray Way (Plasma)
Ray uses **Plasma**, an in-memory object store.
1.  Process A calls `ray.put(obj)`.
2.  Ray writes object to Shared Memory (using Arrow format).
3.  Ray returns `ObjectRef`.
4.  Process A sends `ObjectRef` to Process B (Tiny message).
5.  Process B looks up `ObjectRef`. It gets a **Read-Only Viewer** (Pointer) to the shared memory.
6.  **Result:** 0 copies (on same node). Instant access.

### 3. Argument Handling
*   **Small Args (<100KB):** Copied directly in task metadata.
*   **Large Args (>100KB):** Automatically `ray.put()` into Object Store.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Zero-Copy Demo

#### 📁 `src/03_zero_copy.py`
```python
import ray
import numpy as np
import time

ray.init()

# Create a large array (1GB)
# np.ones(size) * 8 bytes (float64)
N = 100 * 1000 * 1000 // 8 # ~100MB doubles
large_array = np.ones(N)

@ray.remote
def read_array(arr):
    # This function receives a COPY if using multiprocessing
    # But receives a VIEW if using Ray
    start = time.time()
    val = arr[0]
    return time.time() - start

# Case 1: Pass by Value (Implicit put)
# Ray intelligently puts it in store, but let's time it.
start = time.time()
obj_ref = ray.put(large_array)
print(f"Put time: {time.time() - start:.4f}s")

# Case 2: Pass Reference
# Workers just map the memory.
futures = [read_array.remote(obj_ref) for _ in range(5)]
results = ray.get(futures)

print(f"Read times (should be near 0): {results}")

# Verify it is read-only
@ray.remote
def try_mutate(arr):
    try:
        arr[0] = 999
        return "Mutated"
    except ValueError as e:
        return f"failed: {e}"

print(ray.get(try_mutate.remote(obj_ref))) 
# Output: failed: assignment destination is read-only
```

### 👨‍💻 Core Implementation: Memory Management

Objects in Plasma are immutable. To reclaim space, the `ObjectRef` must go out of scope (Reference Counting).

```python
# To force delete
del obj_ref
# Ray's garbage collector will eventually free the shared memory
```

---

## 🔬 Lab Exercise: "Distributed DataFrame"

### Task
Process a Pandas DataFrame.
1.  Create a large DataFrame.
2.  `ray.put(df)`.
3.  Launch 4 tasks. Each task computes the sum of a specific column.
4.  **Note:** Ray converts Pandas to Arrow automatically for storage. This is extremely efficient.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Immutability:** Once an object is `ray.put()`, it cannot be changed. This eliminates race conditions but requires a functional programming style (Input -> New Output).
2.  **Spilling:** If Object Store fills up (default: 30% of RAM), Ray spills objects to Disk (/tmp/ray). This kills performance. Monitor `ray memory`.
3.  **Pattern:** Common pattern involves a "Driver" putting data, and "Workers" reading it.

### API Summary
```bash
ray memory --verbose
```

---

**Day 86 Complete** ✅

*Next: Day 87 - Scheduling & Resources - GPUs, custom resources, and placement groups.*
