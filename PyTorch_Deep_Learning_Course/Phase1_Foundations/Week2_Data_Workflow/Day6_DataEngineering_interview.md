# Day 6: Data Engineering - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: I/O Optimization, Sampling, and Parallelism

### 1. Explain the difference between `Dataset` and `DataLoader`.
**Answer:**
*   **Dataset**: Holds the data (or logic to access it). Implements `__len__` and `__getitem__`.
*   **DataLoader**: Handles the logistics. Batching, Shuffling, Multiprocessing, Pinning memory. It iterates over the Dataset.

### 2. Why does increasing `num_workers` sometimes slow down training?
**Answer:**
*   **Overhead**: Spawning processes takes time.
*   **Context Switching**: If workers > CPU cores, OS spends time switching.
*   **IO Bottleneck**: If disk is slow, adding workers just adds contention.
*   **Memory**: Each worker consumes RAM. Swapping to disk kills performance.

### 3. What is the "Bus Error" in PyTorch DataLoader?
**Answer:**
*   It usually means **Shared Memory** (`/dev/shm`) is full.
*   PyTorch puts tensors in shared memory to pass them between workers and main process.
*   Common in Docker containers (default 64MB). Fix by increasing shm-size.

### 4. How do you handle a dataset that doesn't fit in RAM?
**Answer:**
*   **Lazy Loading**: Load file in `__getitem__`, not `__init__`.
*   **Memory Mapping (`mmap`)**: Map file on disk to virtual memory.
*   **Streaming**: Use `IterableDataset` to read from network/disk sequentially.

### 5. What is the purpose of `collate_fn`?
**Answer:**
*   It takes a list of samples (output of `__getitem__`) and merges them into a batch.
*   Default collate stacks tensors.
*   Custom collate is needed for: Variable length sequences (Padding), Custom objects, Object Detection (List of Bounding Boxes).

### 6. Explain `WeightedRandomSampler`.
**Answer:**
*   Used for Imbalanced Datasets.
*   Instead of sampling uniformly ($p = 1/N$), we assign weights $w_i$ to each sample (usually inverse of class frequency).
*   Ensures each batch has balanced class distribution.

### 7. What is the difference between `shuffle=True` in DataLoader and shuffling the dataset beforehand?
**Answer:**
*   `shuffle=True` reshuffles indices *every epoch*. This is crucial for SGD convergence.
*   Shuffling beforehand (static) means the model sees the same order every epoch, which can lead to overfitting to the order.

### 8. Why is `pin_memory=True` recommended?
**Answer:**
*   It uses Page-Locked RAM.
*   Allows asynchronous DMA transfer to GPU (CPU doesn't wait).
*   Speeds up `inputs.to(device)`.

### 9. How does `IterableDataset` differ from `Dataset` in multi-process loading?
**Answer:**
*   In `Dataset` (Map-style), workers are assigned indices (Worker 1 gets 0, 4, 8...).
*   In `IterableDataset`, every worker gets a copy of the *entire* iterator.
*   You must manually shard the data inside `__iter__` using `get_worker_info()` to prevent workers from returning duplicate data.

### 10. What is "Prefetch Factor"?
**Answer:**
*   Number of batches loaded in advance by each worker.
*   Helps smooth out I/O latency spikes.
*   Default is 2.

### 11. How do you debug a silent hang in DataLoader?
**Answer:**
*   Usually caused by a deadlock in multiprocessing or an error in a worker that isn't propagated.
*   Set `num_workers=0` to run on main thread. If it crashes with a stack trace, you found the bug.

### 12. What is the Global Interpreter Lock (GIL) impact on DataLoading?
**Answer:**
*   GIL prevents threads from running Python bytecode in parallel.
*   That's why we use `multiprocessing` (Processes) instead of `threading`. Processes have separate GILs.

### 13. How do you implement "Dynamic Batching"?
**Answer:**
*   Grouping samples of similar length together to minimize padding.
*   Usually done by a custom `BatchSampler` or `collate_fn`.
*   Common in NLP (Transformers).

### 14. What is `worker_init_fn`?
**Answer:**
*   Function called in each worker on startup.
*   Used to seed random number generators (numpy/random) differently for each worker.
*   Without this, all workers might produce the same random augmentations!

### 15. What is "Copy-on-Write" (CoW) in the context of PyTorch Datasets?
**Answer:**
*   When forking a process, memory is shared until modified.
*   If `__init__` loads a large array, workers share it.
*   But if Python reference counts change (even reading can trigger this), CoW triggers a copy, duplicating memory usage.

### 16. How do you handle "Corrupt Images" in a dataset during training?
**Answer:**
*   In `__getitem__`, wrap load in `try-except`.
*   If error, return `None`.
*   In `collate_fn`, filter out `None` values.
*   Or return a replacement image (black) to keep batch size constant.

### 17. What is WebDataset?
**Answer:**
*   A format and library for high-performance I/O.
*   Stores data in POSIX tar archives (shards).
*   Enables sequential reading at disk speed (500MB/s+), avoiding random seek overhead of millions of small files.

### 18. How does `drop_last=True` affect training?
**Answer:**
*   Drops the last batch if it's smaller than `batch_size`.
*   Useful if your model (or BatchNorm) requires fixed batch size or fails on size 1.

### 19. Can you use a Generator as a Dataset?
**Answer:**
*   Yes, via `IterableDataset`.
*   Useful for infinite datasets (Reinforcement Learning replay buffer, Synthetic data).

### 20. What is the "NVIDIA DALI" library?
**Answer:**
*   Data Loading Library that performs decoding (JPEG) and augmentation on the **GPU**.
*   Offloads CPU. Extremely fast.
