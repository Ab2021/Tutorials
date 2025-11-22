# Day 1: State Backends - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between HashMapStateBackend and RocksDBStateBackend?**
    -   *A*: HashMap stores objects on Heap (fast, GC pressure, size limit). RocksDB stores serialized bytes on Disk/Off-Heap (slower, scalable, no GC).

2.  **Q: How does Flink handle state larger than memory?**
    -   *A*: By using the RocksDB state backend, which spills to disk.

3.  **Q: What is an Incremental Checkpoint?**
    -   *A*: A checkpoint that only persists the changes (diff) since the last checkpoint, rather than the full state.

### Production Challenges
-   **Challenge**: **Long GC Pauses**.
    -   *Scenario*: Using HashMapStateBackend with large windows.
    -   *Fix*: Switch to RocksDB.

-   **Challenge**: **RocksDB High CPU**.
    -   *Cause*: Heavy serialization or aggressive compaction.
    -   *Fix*: Optimize data types (avoid generic Objects), tune compaction threads.

### Troubleshooting Scenarios
**Scenario**: Checkpoint fails with "Size exceeded".
-   *Cause*: State is too large for the target storage or timeout.
-   *Fix*: Enable incremental checkpoints, increase timeout, or check for state leaks (keys never deleted).
