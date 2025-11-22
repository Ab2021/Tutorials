# Day 5: Exactly-Once - Deep Dive

## Deep Dive & Internals

### Kafka Transactional Protocol
Flink acts as a Kafka Producer.
-   **Transaction ID**: Derived from `JobName + OperatorID`. Must be consistent across restarts.
-   **Zombies**: If a Flink task crashes, the new task must "fence" the old zombie transaction to prevent data corruption. Kafka handles this via Epochs.

### The "Read-Committed" Isolation
Downstream consumers MUST be configured with `isolation.level=read_committed`.
-   Otherwise, they will see "open" (uncommitted) transactions, breaking exactly-once.

### Advanced Reasoning
**Latency Trade-off**
E2E Exactly-Once adds latency.
-   Data is only visible downstream after the checkpoint completes (Commit).
-   Latency = `checkpoint.interval` + processing time.
-   If you need sub-second latency, you might have to accept At-Least-Once.

### Performance Implications
-   **Transaction Overhead**: Kafka transactions have overhead. Don't set checkpoint interval too low (e.g., < 1s).
