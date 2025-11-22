# Day 5: Reliability & Durability - Deep Dive

## Deep Dive & Internals

### The High Watermark (HW)
The HW is the offset of the *last message that was successfully replicated to all ISRs*.
-   **Consumers only see up to HW**. They cannot read uncommitted data.
-   This prevents "phantom reads" (reading data that is later lost due to leader failure).

### Leader Epochs
Used to prevent data loss during tricky failure scenarios (like a follower becoming leader, then the old leader coming back with divergent logs).
-   **Epoch**: A monotonic counter increased every time a new leader is elected.
-   Brokers truncate their logs to the point where they diverged from the *current* leader's epoch.

### Idempotent Producer
-   **Sequence Numbers**: Each message has a (ProducerID, SequenceNumber).
-   **De-duplication**: The broker keeps track of the last SequenceNumber. If it receives a duplicate (due to network retry), it silently drops it and acks.
-   **Overhead**: Negligible. Always enable it (`enable.idempotence=true`).

### Advanced Reasoning
**Why not synchronous replication to ALL replicas?**
-   If you have 3 replicas and wait for ALL 3, then 1 slow/dead node stops the whole cluster.
-   **ISR** is the compromise: Wait only for the *healthy* ones. If a node is slow, kick it out of ISR so it doesn't block writes.

### Performance Implications
-   `acks=all`: Increases latency (must wait for network round-trip to followers).
-   `compression`: Reduces network bandwidth and disk usage, improving effective durability (faster replication).
