# Day 5: Reliability & Durability

## Core Concepts & Theory

### Replication
Kafka replicates partitions across multiple brokers for fault tolerance.
-   **Leader**: The replica that handles all reads and writes.
-   **Follower**: Passive replicas that fetch data from the leader to stay in sync.
-   **ISR (In-Sync Replicas)**: The set of replicas that are currently caught up with the leader.

### Acknowledgements (acks)
Producers can choose their durability level:
-   `acks=0`: Fire and forget. Fastest, least safe.
-   `acks=1`: Leader acknowledges. Safe from follower failure, not leader failure.
-   `acks=all`: All ISRs acknowledge. Strongest durability.

### Min.Insync.Replicas
This config defines the minimum number of replicas that must acknowledge a write for it to be considered successful when `acks=all`.
-   If `min.insync.replicas=2` and only 1 replica is alive, the broker rejects the write.

### Architectural Reasoning
**Consistency vs. Availability (CAP Theorem)**
Kafka defaults to Availability (AP) but can be tuned for Consistency (CP).
-   `acks=all` + `min.insync.replicas=2` favors Consistency.
-   `acks=1` favors Availability and Latency.

### Key Components
-   **Replication Factor**: Total copies of data (usually 3).
-   **ISR**: The "healthy" replicas.
-   **High Watermark**: The offset up to which all ISRs have replicated. Consumers can only read up to here.
