# Day 3: Topics, Partitions, Segments - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Kafka find a message by offset?**
    -   *A*: It uses the sparse `.index` file to find the nearest physical position, then scans forward.

2.  **Q: What is Log Compaction used for?**
    -   *A*: It's used for restoring state (e.g., KTable, database CDC). We only care about the *latest* state of a key, not the history.

3.  **Q: What happens if you have too many partitions?**
    -   *A*: High unavailability during failover (leader election takes time), high memory usage (metadata), and high open file handles.

### Production Challenges
-   **Challenge**: **"Too many open files"**.
    -   *Scenario*: Broker crashes because it hit the OS limit on file descriptors.
    -   *Fix*: Increase `ulimit -n`. Reduce retention period or increase segment size.

-   **Challenge**: **Unbalanced Partitions**.
    -   *Scenario*: One partition is 1TB, others are 1GB.
    -   *Cause*: Poor partition key choice (Data Skew).
    -   *Fix*: Fix the producer's partitioning logic.

### Troubleshooting Scenarios
**Scenario**: Disk usage is not going down despite low retention.
-   **Check**: Is `log.cleanup.policy=compact`? Compaction doesn't delete old data by time immediately; it waits for the cleaner thread.
-   **Check**: Are there "stray" partitions that are not being deleted?
