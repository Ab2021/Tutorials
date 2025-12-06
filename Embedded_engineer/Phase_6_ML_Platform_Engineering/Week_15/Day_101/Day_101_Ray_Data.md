# Day 101: The Firehose: Ray Data Pipelines
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** Your GPUs compute faster than `DataLoader` can read from disk. **Ray Data** decouples I/O from Compute, creating a dedicated, distributed preprocessing cluster that streams batch to the Trainer.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the architectural difference between "Loader in Trainer" (PyTorch) vs "Loader as Service" (Ray Data).
2.  **Use** `ray.data.read_*` to ingest Parquet/CSV/Images lazily.
3.  **Build** a transformation pipeline (Resize -> Normalize) using `.map_batches()`.
4.  **Consume** data in the training loop using `iter_torch_batches()`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install "ray[data]" pandas`.

---

## 📖 Theoretical Foundation

### 1. Unified vs Disaggregated
*   **Unified (PyTorch):** Every training worker (GPU node) launches multiprocessing workers to read disk.
    *   *Problem:* GPU node CPUs are often weak. Disk I/O saturates. Random Augmentation steals cycles.
*   **Disaggregated (Ray Data):**
    *   CPU Nodes: Read Disk -> Preprocess -> Store in Plasma (Object Store).
    *   GPU Nodes: Read from Plasma (Zero Copy / Network Stream).
    *   *Result:* GPU stays at 100% utilization.

### 2. Streaming Execution
Ray Data doesn't materialize the whole dataset. It uses **Streaming Backpressure**.
*   If GPU is slow, Reader pauses.
*   If GPU is fast, Reader scales up tasks.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Data Pipeline

#### 📁 `src/02_ray_data.py`
```python
import ray
import numpy as np

# 1. Create Data (Lazy)
# Imagine 1000 parquet files
ds = ray.data.range(10000) 

# 2. Add Transformations
def preprocess(batch):
    # Simulate augmentation (e.g., resize image)
    # Input is a Dict of numpy arrays
    data = batch["id"]
    return {"id": data, "augmented": data * 2}

# Transform in batches (Vectorized)
# Compute Strategy determines usage (e.g., use 4 CPU workers)
ds = ds.map_batches(preprocess, batch_size=128)

# 3. Split for Training Workers
# In Ray Train, the trainer handles splitting automatically if you pass the dataset.
# But manually:
# ds_list = ds.split(n=2)
```

### 👨‍💻 Core Implementation: Integration with Trainer

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    # 1. Get the dataset shard assigned to this worker
    train_ds = ray.train.get_dataset_shard("train")
    
    # 2. Iterate (Streaming)
    # prefetch_batches ensures next batch is ready on GPU
    batch_iterator = train_ds.iter_torch_batches(
        batch_size=32, 
        prefetch_batches=2
    )
    
    for batch in batch_iterator:
        # batch is a Dict[str, torch.Tensor] already on CPU
        # Move to GPU
        x = batch["augmented"].cuda()
        # forward pass...

# Driver
scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    datasets={"train": ds}, # Pass the lazy dataset here
    dataset_config=ray.train.DataConfig(
        execution_options=ray.data.ExecutionOptions(resource_limits={"CPU": 4})
    )
)
```

---

## 🔬 Lab Exercise: "Bottleneck Analysis"

### Task
Measure Throughput.
1.  Add `time.sleep(0.01)` in `preprocess`.
2.  Run Training.
3.  Check Ray Dashboard -> "Data" tab.
4.  **Observation:** You see "Map Batches" tasks running.
5.  If "Wait on Data" time in Loop is high, increase `resource_limits={"CPU": N}` for the dataset configuration. This allocates MORE CPU workers to generate data faster than the GPU consumes it.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Heterogeneous Scaling:** You can have 2 GPU nodes and 20 CPU nodes. Ray Data uses the 20 CPU nodes to feed the 2 GPU nodes. Standard PyTorch `DataLoader` cannot do this easily.
2.  **Epochs:** Ray Data streams are **single-pass** by default. For multiple epochs, you iterate the shard again (re-reading, or use `.materialize()` if it fits in memory).
3.  **Formats:** Native support for Parquet, CSV, JSON, TFRecord, and Numpy.

### API Summary
```python
ds.map_batches()
ray.train.get_dataset_shard()
```

---

**Day 101 Complete** ✅

*Next: Day 102 - Checkpointing & Fault Tolerance - Saving state without stopping.*
