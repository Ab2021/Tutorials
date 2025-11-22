# Day 5: Exactly-Once - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: Can you achieve Exactly-Once with a non-transactional sink?**
    -   *A*: Only if the sink is Idempotent (e.g., KV store). If it's append-only (like a log) and not transactional, you will get duplicates (At-Least-Once).

2.  **Q: What happens to open transactions if the Flink job fails?**
    -   *A*: Flink recovers, aborts the old transactions (or lets them time out), and replays from the last checkpoint.

3.  **Q: Why do consumers need `read_committed`?**
    -   *A*: To ignore messages that are part of an aborted transaction or a transaction that is still in progress.

### Production Challenges
-   **Challenge**: **Data not visible**.
    -   *Scenario*: Pipeline running, but downstream sees nothing.
    -   *Cause*: `isolation.level=read_committed` and Checkpoints are failing (so nothing is ever committed).
    -   *Fix*: Fix checkpoints.

### Troubleshooting Scenarios
**Scenario**: `ProducerFencedException`.
-   *Cause*: Multiple producers with the same Transactional ID. Usually caused by a zombie task or misconfiguration.
