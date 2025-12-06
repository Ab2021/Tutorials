# Day 125: Tuning the Engine: Ray Data Performance
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 18: Ray Data & Pipelines

---

> **🎯 Focus Area:** "Out of Memory" is the enemy of Big Data. Ray Data uses **Streaming Execution** to process datasets larger than RAM, but you must tune the backpressure knobs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** memory pressure using the Ray Dashboard and CLI.
2.  **Configure** Object Spilling to handle overflows.
3.  **Tune** `ExecutionOptions` to limit active CPU tasks.
4.  **Optimize** Batch Size for maximum throughput without OOM.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Bottleneck Theory
*   **I/O Bound:** Reading files is slow. Increase parallelism?
    *   Yes, but too many reads = OOM (Object Store full).
*   **Compute Bound:** Transformation is slow.
    *   Ray automatically scales up tasks if resources exist.
*   **Backpressure:** Ray limits the number of "in-flight" blocks. If the GPU is stuck, Ray stops reading files.

### 2. Spilling
If Object Store > Limit, Ray writes LRU objects to `/tmp/ray/spill`.
This is tragic for performance. Avoid it by reducing parallelism or batch size.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Controlling Parallelism

#### 📁 `src/06_tuning.py`
```python
import ray
import time
import numpy as np

# 1. Create Heavy Data
# 1000 partitions. Each partition takes 100MB RAM.
ds = ray.data.range(1000).map(lambda x: {"data": np.zeros((1024, 1024))})

# 2. Slow Consumer
def slow_consume(batch):
    time.sleep(1)
    return batch

# 3. Default Execution (Might OOM if it reads too fast)
# Ray generally handles this well now with streaming.

# 4. Explicit Limits
ctx = ray.data.DataContext.get_current()
ctx.execution_options.resource_limits.cpu = 2 
# Meaning: Only allow 2 CPU tasks (reads/maps) to run concurrently.
# This strictly limits memory usage.

ds.map_batches(slow_consume).take_all()
```

### 👨‍💻 Core Implementation: Monitoring

Run this in terminal while script runs:
```bash
ray memory --verbose
```
Look for "Spilled" bytes. Ideally 0.

---

## 🔬 Lab Exercise: "The Straw"

### Task
Break it.
1.  Set `resource_limits.cpu = 100`.
2.  Run the heavy data script.
3.  **Observation:** Valid Object Store memory Usage sky rockets.
4.  If it hits 100% of Object Store limit, you see "Spilling IO" warnings in logs. Throughput drops 10x.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Preserve Order:** `ds.map_batches(..., preserve_order=True)` forces serialization. Parallelism drops. Avoid if possible.
2.  **Small Objects:** A dataset of 10 million small strings (10 bytes each) causes overhead issues. Consolidate them into lists/batches inside the blocks.
3.  **Dashboard:** The "Data" tab explains exactly *why* it is slow (e.g., "Waiting for CPU").

### API Summary
```python
ray.data.DataContext.get_current().execution_options
```

---

**Day 125 Complete** ✅

*Next: Day 126 - Week 18 Review & Project - The Petabyte Sorter.*
