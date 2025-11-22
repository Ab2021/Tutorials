# Day 2: Checkpointing - Deep Dive

## Deep Dive & Internals

### Unaligned Checkpoints
In high-backpressure scenarios, barrier alignment takes too long (barriers are stuck in queues).
-   **Unaligned**: Snapshot the *inflight data* (buffers) along with the state.
-   **Pros**: Checkpoints succeed even under heavy load.
-   **Cons**: Larger checkpoint size (storing buffers).

### State Recovery Process
1.  **Restart**: Job fails, all tasks restart.
2.  **Load Metadata**: JobManager reads the latest valid checkpoint metadata.
3.  **Restore State**: TaskManagers download their assigned state chunks from S3.
4.  **Resume**: Processing starts from the checkpoint barrier.

### Advanced Reasoning
**Savepoint vs Checkpoint**
-   **Checkpoint**: "Implementation detail". Optimized for speed. Might depend on specific file paths. Not guaranteed to be portable.
-   **Savepoint**: "Logical backup". Optimized for portability. Can be used to upgrade Flink version or change parallelism.

### Performance Implications
-   **Copy-On-Write**: HashMapBackend uses COW to snapshot without blocking.
-   **Async Upload**: The heavy lifting (uploading to S3) happens in the background.
