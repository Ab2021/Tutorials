# Day 186: The Overflow: Memory Leak Hunting
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** Your training job runs fine for 10 hours, then crash loops with `OOM Killed`. This slow death indicates a Memory Leak. Finding it requires tracing references in Python and Allocations in C++.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Visualize** PyTorch memory allocations using the `torch.cuda.memory._record_memory_history()` snapshot.
2.  **Debug** Python Circular References using `gc` and `objgraph`.
3.  **Profile** native C++ Extension leaks using `memray`.
4.  **Mitigate** Memory Fragmentation by tuning `max_split_size_mb`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install memray objgraph torch`.

---

## 📖 Theoretical Foundation

### 1. Types of Leaks
*   **Python Reference Cycle:** Object A refs B, B refs A. RefCount never hits 0. GC *should* collect it, but if `__del__` is defined (legacy) or logic prevents it, it stays.
*   **Accumulated History:** `loss += current_loss` keeps the entire Computational Graph of every batch in memory forever. Correct: `loss += current_loss.item()`.
*   **C++ Leak:** Dataloader using OpenCV/Pillow has a C++ leak. Python GC doesn't see it. `RSS` memory grows, but Python heap stays flat.

### 2. Fragmentation
Memory is swiss cheese.
You have 10GB free total, but largest contiguous block is 100MB.
Requesting 200MB fails with OOM.
**Solutions:** Pre-allocate, or restart process periodically.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: PyTorch Memory Snapshot

Dump the state of the CUDA allocator.

#### 📁 `src/snapshot_debug.py`
```python
import torch
import pickle

def train_step():
    # Simulate heavy workload
    t = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()

# 1. Start Recording
torch.cuda.memory._record_memory_history(max_entries=100000)

try:
    for i in range(10):
        train_step()
except RuntimeError:
    print("OOM Detected!")

# 2. Dump Snapshot
print("Saving snapshot to trace.pickle")
timestamp = torch.cuda.memory._snapshot()
with open("trace.pickle", "wb") as f:
    pickle.dump(timestamp, f)

# 3. Visualization
# Go to https://pytorch.org/memory_viz -> Upload trace.pickle
# You will see a timeline of every malloc/free.
```

### 👨‍💻 Core Implementation: Finding Ref Cycles

Why isn't my Model getting deleted?

#### 📁 `src/ref_cycle_hunt.py`
```python
import objgraph
import gc
import torch
import weakref

class BadNode:
    def __init__(self):
        self.child = None
        
    def add_child(self, node):
        self.child = node
        # CIRCULAR: Child points back to parent
        node.parent = self 

def create_leak():
    parent = BadNode()
    child = BadNode()
    parent.add_child(child)
    return parent

# 1. Create Garbage
create_leak()
create_leak()

# 2. Force Collection
gc.collect()

# 3. Inspect
print("Most common types in memory:")
objgraph.show_most_common_types(limit=5)

# 4. Find Backrefs (Who is holding this object?)
# Identify random 'BadNode' lingering in memory
leaked_objs = objgraph.by_type('BadNode')
if leaked_objs:
    print("Found leaked BadNode!")
    # Draw graph of references
    objgraph.show_backrefs(leaked_objs[0], filename="leak_graph.png")
```

### 👨‍💻 Core Implementation: Memray (Native Profiling)

Run your script with memray wrapper.

```bash
# Run
memray run -o output.bin my_script.py

# Report
memray flamegraph output.bin
# Opens HTML. Look for tall towers (Stacks allocating lots of RAM).
```

---

## 🔬 Lab Exercise: "The Accumulator"

### Task
Fix the classic PyTorch bug.
1.  **Bug Code:**
    ```python
    total_loss = 0
    for data, target in loader:
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        total_loss += loss # BUG: Stores Graph!
    ```
2.  **Observation:** Memory grows linearly with steps until OOM.
3.  **Fix:** `total_loss += loss.item()`.
4.  **Verification:** Monitor `RSS` (Resident Set Size). It should stay flat.

---

## 📖 Advanced Theory: Allocator Settings
PyTorch uses a caching allocator (cub/caching_allocator).
Sometimes it caches too aggressively.
**Environment Variable:** `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
Tells PyTorch: "Don't split blocks larger than 128MB. Release them to OS/Driver earlier."
Prevents fragmentation where large requests fail because memory is sliced into tiny cached blocks.

---

## 📝 Daily Summary

### Key Takeaways
1.  **RSS vs VMS:** Ignore Virtual Memory (VMS) on Linux; it includes shared libs and unallocated maps. Look at RSS (Resident Set Size) for real physical RAM usage.
2.  **Cuda Malloc Retries:** If PyTorch fails to malloc, it empties its cache and retries. This is expensive (causes stutter). Ideally, cache usage should be stable.
3.  **Dataloaders:** Workers (multiprocessing) duplicate some memory (Copy-on-Write). If you write to a global variable in the worker, the Whole variable is copied. Keep workers stateless.

### API Summary
```python
gc.get_objects()
torch.cuda.empty_cache() # Don't use in training loop! Slow.
```

---

**Day 186 Complete** ✅

*Next: Day 187 - Log Analytics at Scale (Loki).*
