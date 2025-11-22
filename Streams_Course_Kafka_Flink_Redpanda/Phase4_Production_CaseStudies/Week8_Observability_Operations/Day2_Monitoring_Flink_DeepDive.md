# Day 2: Monitoring Flink - Deep Dive

## Deep Dive & Internals

### Backpressure Monitoring
How does Flink know it's backpressured?
1.  **Old Way (Stack Trace Sampling)**: JobManager periodically triggers thread dumps on TaskManagers to see if they are stuck in `requestBuffer`. High overhead.
2.  **New Way (Credit-Based)**: TaskManagers report `outPoolUsage` (how many output buffers are full). If buffers are full, it cannot send data -> Backpressure. Zero overhead.

### Checkpoint Monitoring
Checkpoint metrics reveal the health of your state backend.
-   **Sync Duration**: Time to snapshot state in memory. If high -> CPU bottleneck or huge state object.
-   **Async Duration**: Time to upload to S3/DFS. If high -> Network/Storage bottleneck.
-   **Alignment Time**: Time waiting for barriers. If high -> Skew or Backpressure.

### Memory Monitoring
Flink manages its own off-heap memory (Managed Memory).
-   **Heap**: User code objects.
-   **Off-Heap**: Network buffers, RocksDB native memory.
-   **Metaspace**: Class metadata.
**OOM Debugging**:
-   `Heap Space OOM`: User code memory leak.
-   `Direct Buffer OOM`: Network buffer leak or RocksDB growing too large.

### Performance Implications
-   **User Metrics**: Don't create a new `Counter` for every user ID (Cardinality explosion!).
-   **Tagging**: Flink adds tags (`job_name`, `task_name`, `subtask_index`). Ensure your TSDB supports high cardinality if you have many jobs.
