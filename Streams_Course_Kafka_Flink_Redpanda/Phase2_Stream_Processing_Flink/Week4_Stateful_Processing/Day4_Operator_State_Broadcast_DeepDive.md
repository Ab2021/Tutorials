# Day 4: Operator State - Deep Dive

## Deep Dive & Internals

### Checkpointing Operator State
-   **Snapshot**: Each task snapshots its local list/map.
-   **Restore**:
    -   **Even-Split**: (Default for ListState). The global list is shuffled and distributed evenly.
    -   **Union**: (For Kafka Offsets). Every task gets the *full* list of all offsets, then decides which ones it owns.

### Broadcast State Consistency
-   Flink guarantees that the broadcast element is processed by *all* parallel instances.
-   However, there is no "Global Consensus". Instance A might see Rule V2 at time T, while Instance B sees it at T+1. Order is preserved per-stream, but cross-stream synchronization depends on watermarks (if used).

### Advanced Reasoning
**Kafka Source Offsets**
The Kafka Source uses **Operator State (ListState)** to store the offsets `(Topic, Partition, Offset)`.
-   When parallelism increases, the list of partitions is redistributed among the new tasks.

### Performance Implications
-   **Broadcast Size**: Don't broadcast gigabytes of data. It is replicated in memory on *every* TaskManager. It will blow up the heap. Keep broadcast state small (MBs).
