# Day 1: Skew - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you detect skew in a running Flink job?**
    -   *A*: Check the Flink Web UI. Look for **Backpressure** on specific subtasks. Check `numRecordsIn` per subtask. If Subtask 0 has 100x more records than Subtask 1, it's skew.

2.  **Q: Can you handle skew without changing the code?**
    -   *A*: Sometimes. Flink SQL has `table.exec.skew-join.enabled` (adaptive skew handling). But usually, you need explicit salting.

3.  **Q: What is the downside of Salting?**
    -   *A*: Correctness complexity. You can't rely on global ordering anymore. And you need a second aggregation step.

### Production Challenges
-   **Challenge**: **Dynamic Skew**.
    -   *Scenario*: A key becomes hot suddenly (Breaking News).
    -   *Fix*: Adaptive Salting. Detect hot keys at runtime and only salt those. (Complex to implement).

-   **Challenge**: **Checkpoint Timeout**.
    -   *Scenario*: Skewed task takes too long to process barriers. Checkpoint fails.
    -   *Fix*: Unaligned Checkpoints (Flink 1.11+). Allows barriers to jump over data.
