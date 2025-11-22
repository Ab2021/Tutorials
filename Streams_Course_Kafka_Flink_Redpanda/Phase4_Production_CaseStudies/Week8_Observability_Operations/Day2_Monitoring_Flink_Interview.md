# Day 2: Monitoring Flink - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you detect data skew using metrics?**
    -   *A*: Compare `numRecordsIn` across subtasks. If Subtask 0 has 1M records and Subtask 1 has 100 records, you have skew.

2.  **Q: What does it mean if `lastCheckpointDuration` is close to `checkpointInterval`?**
    -   *A*: Dangerous. The job spends all its time checkpointing. If it exceeds the interval, checkpoints will start failing or skipping.

3.  **Q: How do you monitor RocksDB specifically?**
    -   *A*: Enable `state.backend.rocksdb.metrics.enable`. Monitor `block-cache-usage`, `memtable-size`, and `estimate-num-keys`.

### Production Challenges
-   **Challenge**: **"Silent" Failure**.
    -   *Scenario*: Job is running (RUNNING status), but processing 0 records.
    -   *Fix*: Alert on `numRecordsOutPerSecond == 0` for > 5 mins.

-   **Challenge**: **GC Pauses**.
    -   *Scenario*: Job pauses for 10s every minute. Throughput drops.
    -   *Fix*: Monitor `GarbageCollector.CollectionTime`. Tune JVM (G1GC), increase Heap, or reduce object creation rate.

### Troubleshooting Scenarios
**Scenario**: `CheckpointExpiredException`.
-   *Cause*: Checkpoint took longer than timeout (10 mins).
-   *Fix*: Check `AsyncDuration` (Storage slow?) and `AlignmentTime` (Backpressure?). Increase timeout or optimize state.
